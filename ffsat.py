# Uncomment if you need to inspect precise memory allocation/movements
#import os
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable pre-allocation
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform-specific allocator

import typer
import argparse
from time import perf_counter as time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Optional as Opt, NamedTuple, Iterable, Any
from jax.typing import ArrayLike as Array

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import ProjectedGradient#, ProximalGradient
from jaxopt.projection import projection_box

from boolean_whf import ClauseGroup, ApproxLenClauses, class_map, Objective, ClauseArrays
from sat_loader import Formula

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_disable_jit", True)


class Validators(NamedTuple):
    xor: Objective
    cnf: Objective
    eo: Objective
    nae: Objective
    card: Objective
    amo: Objective


def valid_choice(value: str) -> int:
    choice_map = {"1": "full", "2": "types", "3": "lens"}
    if int(value) not in {1, 2, 3}:
        raise typer.BadParameter("Choice must be 1, 2, or 3")
    return choice_map[value]


def preprocess_all(sat: Formula, mode: Opt[int], threshold: int = 0) -> tuple[tuple[Objective, ...], Validators]:
    def approx_mem() -> float:
        pass

    clause_grps: dict[str, ClauseGroup] = {}
    if mode is not None and mode > 0:
        choice = valid_choice(str(mode))
    else:
        print("Please see the following clause type and clause length breakdowns and select an option:")
        print(
            "Types count:\n\t",
            [f"{x}: {len(y)} ({sat.stats[x]})" for (x, y) in sat.clauses_type.items() if y],
        )
        print("Lengths count\n\t", [f"{x}: {len(y)}" for (x, y) in sat.clauses_len.items()])
        print(
            "\tOptions:\n"
            + "\t\t1: Full combine. Use single monolithlic array with all clauses appropriately padded\n"
            + "\t\t2: By type. Separate padded array for each clause type\n"
            + "\t\t3: By length. Separate (possibly minor padding) for each clause length (or length cluster)"
        )
        choice = typer.prompt("Options", type=valid_choice, default="2")

    n_var = sat.n_var
    # We need the breakdown by clause type anyway for quick validation.
    for c_type, c_list in sat.clauses_type.items():
        if c_list:
            # Clauses present, do FFTs if user partition choice was by type.
            clause_grps[c_type] = class_map[c_type](c_list, n_var, do_fft=(choice == "types"))
        else:
            clause_grps[c_type] = None

    match choice:
        case "full":
            clause_grps["full"] = ApproxLenClauses(sat.clauses_all, n_var, do_fft=True)
        case "lens":
            # thresh = 0
            # TODO: Implement clustering with some threshold?
            for c_len, c_list in sat.clauses_len.items():
                clause_grps[c_len] = ApproxLenClauses(c_list, n_var, do_fft=True)
        case _:
            # type already proesssed above.
            pass

    # Process groups!
    def process_groups(clause_grps: dict[str, ClauseGroup], max_workers: Opt[int] = None):
        def process(grp: ClauseGroup):
            grp.process()

        with ThreadPoolExecutor(max_workers=max_workers) as tpool:
            tasks = [tpool.submit(process, grp) for grp in clause_grps.values() if grp]
            for task in tasks:
                task.result()

    process_groups(clause_grps, max_workers=min(len(clause_grps), 8))

    empty_Clause = ClauseArrays(sparse=jnp.zeros((0, 0, n_var), dtype=int))
    empty_Validation = Objective(empty_Clause, None, None, jnp.zeros((0, 0), dtype=int))
    objectives = []
    validation = {}

    for grp_type, grp in clause_grps.items():
        if not grp:
            # No clause set, so in clause_grps for validation, add the empty.
            validation[grp_type] = empty_Validation
            continue

        # Valid clause set
        objective = grp.get()
        if grp_type in class_map:
            # Required for validation
            validation[grp_type] = objective
        if choice == "types" or (grp_type not in class_map):
            # Optimisation objective
            objectives.append(objective)

    objectives = tuple(sorted(objectives, key=lambda x: x.clauses.lits.shape[-1]))
    validation = Validators(**validation)
    return objectives, validation


@jax.jit
def fval_one(x: Array, objective: Objective, weight: Array) -> Array:
    lits, sign, _, _, _ = objective.clauses
    dft, idft = objective.ffts
    forward_mask = objective.forward_mask
    assignment = sign * x[lits]

    # Evaluate ESPs
    prod = jnp.prod(dft + assignment[:, None, :], axis=2, where=forward_mask)
    clause_eval = jnp.sum(idft * prod, axis=1).real
    cost = weight * clause_eval
    return jnp.sum(cost)


@jax.jit
def fval_one_instrumented_sparse(x: Array, objective: Objective, weight: Array) -> Array:
    _, sign, _, onehot, _ = objective.clauses
    dft, idft = objective.ffts
    forward_mask = objective.forward_mask

    # Use matrix multiplication instead of indexing
    xlits = jnp.einsum("v,clv->cl", x, onehot)  #
    assignment = sign * xlits  # x[lits]

    # Evaluate ESPs
    prod = jnp.prod(dft + assignment[:, None, :], axis=2, where=forward_mask)
    clause_eval = jnp.sum(idft * prod, axis=1).real
    cost = weight * clause_eval
    return jnp.sum(cost)


@jax.jit
def fval(x: Array, objs: tuple[Objective, ...], weight: Array) -> Array:
    # TODO: Fix weight - keeping at 1 for now. #weight[i] eventually
    costs = []
    for objective in objs:
        costs.append(fval_one(x, objective, weight))
        # costs.append(fval_one_instrumented_sparse(x, objective, weight))
    return jnp.sum(jnp.array(costs))


@jax.jit
def verify_sparse(
    x0: Array, xor: Objective, cnf: Objective, eo: Objective, nae: Objective, card: Objective, amo: Objective
) -> dict[str, Array] | Array:
    # Uses onehot encodings with einsum, instead of x[addr]. Would be better if BCOO was better supported?
    @jax.jit
    def unsats_xor(x0: Array, sparse: Array, sign: Array, mask: Array):
        assign = sign * jnp.einsum("v,clv->cl", x0, sparse, optimize=True)
        # Even count (%2==0) of True (<0) means the XOR is UNSAT.
        unsat = jnp.sum(assign < 0, axis=1, where=mask) % 2 == 0
        return unsat

    @jax.jit
    def unsats_cnf(x0: Array, sparse: Array, sign: Array, mask: Array):
        assign = sign * jnp.einsum("v,clv->cl", x0, sparse)
        unsat = jnp.min(assign, axis=1, where=mask, initial=float("inf")) > 0
        return unsat

    @jax.jit
    def unsats_eo(x0: Array, sparse: Array, sign: Array, mask: Array):
        assign = sign * jnp.einsum("v,clv->cl", x0, sparse)
        unsat = jnp.sum(assign < 0, axis=1, where=mask) != 1
        return unsat

    @jax.jit
    def unsats_nae(x0: Array, sparse: Array, sign: Array, mask: Array):
        assign = sign * jnp.einsum("v,clv->cl", x0, sparse)
        has_true = jnp.min(assign, axis=1, where=mask, initial=float("inf")) < 0
        has_false = jnp.max(assign, axis=1, where=mask, initial=float("-inf")) > 0
        unsat = jnp.logical_not(jnp.logical_and(has_true, has_false))
        return unsat

    @jax.jit
    def unsats_card(x0: Array, sparse: Array, sign: Array, mask: Array, cards: Array):
        assign = sign * jnp.einsum("v,clv->cl", x0, sparse)
        unsat = jnp.sum(assign < 0, axis=1, where=mask) < cards
        return unsat

    @jax.jit
    def unsats_amo(x0: Array, sparse: Array, sign: Array, mask: Array):
        assign = sign * jnp.einsum("v,clv->cl", x0, sparse)
        unsat = jnp.sum(assign < 0, axis=1, where=mask) > 1
        return unsat

    # unsat = {
    #     "xor": jnp.sum(unsats_xor(x0, xor.clauses.sparse, xor.clauses.sign, xor.clauses.mask)),
    #     "cnf": jnp.sum(unsats_cnf(x0, cnf.clauses.sparse, cnf.clauses.sign, cnf.clauses.mask)),
    #     "eo": jnp.sum(unsats_eo(x0, eo.clauses.sparse, eo.clauses.sign, eo.clauses.mask)),
    #     "nae": jnp.sum(unsats_nae(x0, nae.clauses.sparse, nae.clauses.sign, nae.clauses.mask)),
    #     "card": jnp.sum(unsats_card(x0, card.clauses.sparse, card.clauses.sign, card.clauses.mask, card.cards)),
    #     "amo": jnp.sum(unsats_amo(x0, amo.clauses.sparse, amo.clauses.sign, amo.clauses.mask)),
    # }
    # unsat = (
    #     (jnp.sum(unsats_xor(x0, xor.clauses.sparse, xor.clauses.sign, xor.clauses.mask)))
    #     + (jnp.sum(unsats_cnf(x0, cnf.clauses.sparse, cnf.clauses.sign, cnf.clauses.mask)))
    #     + (jnp.sum(unsats_eo(x0, eo.clauses.sparse, eo.clauses.sign, eo.clauses.mask)))
    #     + (jnp.sum(unsats_nae(x0, nae.clauses.sparse, nae.clauses.sign, nae.clauses.mask)))
    #     + (jnp.sum(unsats_card(x0, card.clauses.sparse, card.clauses.sign, card.clauses.mask, card.cards)))
    #     + (jnp.sum(unsats_amo(x0, amo.clauses.sparse, amo.clauses.sign, amo.clauses.mask)))
    # )
    unsat = jnp.array(
        [
            (jnp.sum(unsats_xor(x0, xor.clauses.sparse, xor.clauses.sign, xor.clauses.mask))),
            (jnp.sum(unsats_cnf(x0, cnf.clauses.sparse, cnf.clauses.sign, cnf.clauses.mask))),
            (jnp.sum(unsats_eo(x0, eo.clauses.sparse, eo.clauses.sign, eo.clauses.mask))),
            (jnp.sum(unsats_nae(x0, nae.clauses.sparse, nae.clauses.sign, nae.clauses.mask))),
            (jnp.sum(unsats_card(x0, card.clauses.sparse, card.clauses.sign, card.clauses.mask, card.cards))),
            (jnp.sum(unsats_amo(x0, amo.clauses.sparse, amo.clauses.sign, amo.clauses.mask))),
        ]
    )
    return unsat


@jax.jit
def verify(
    x0: Array, xor: Objective, cnf: Objective, eo: Objective, nae: Objective, card: Objective, amo: Objective
) -> dict[str, Array] | Array:
    @jax.jit
    def unsats_xor(x0: Array, lits: Array, sign: Array, mask: Array):
        assign = sign * x0[lits]
        # Even count (%2==0) of True (<0) means the XOR is UNSAT.
        unsat = jnp.sum(assign < 0, axis=1, where=mask) % 2 == 0
        return unsat

    @jax.jit
    def unsats_cnf(x0: Array, lits: Array, sign: Array, mask: Array):
        assign = sign * x0[lits]
        unsat = jnp.min(assign, axis=1, where=mask, initial=float("inf")) > 0
        return unsat

    @jax.jit
    def unsats_eo(x0: Array, lits: Array, sign: Array, mask: Array):
        assign = sign * x0[lits]
        unsat = jnp.sum(assign < 0, axis=1, where=mask) != 1
        # jax.debug.print("{}", jnp.sum(unsat))
        return unsat

    @jax.jit
    def unsats_nae(x0: Array, lits: Array, sign: Array, mask: Array):
        assign = sign * x0[lits]
        has_true = jnp.min(assign, axis=1, where=mask, initial=float("inf")) < 0
        has_false = jnp.max(assign, axis=1, where=mask, initial=float("-inf")) > 0
        unsat = jnp.logical_not(jnp.logical_and(has_true, has_false))
        return unsat

    @jax.jit
    def unsats_card(x0: Array, lits: Array, sign: Array, mask: Array, cards: Array):
        assign = sign * x0[lits]
        # unsat = jnp.sum(assign < 0, axis=1, where=mask) < cards
        sat_count = jnp.sum(assign < 0, axis=1, where=mask)
        unsat = jnp.where(cards < 0, sat_count >= jnp.abs(cards), sat_count < cards)
        # jax.debug.print("{}", jnp.sum(unsat))
        return unsat

    @jax.jit
    def unsats_amo(x0: Array, lits: Array, sign: Array, mask: Array):
        assign = sign * x0[lits]
        unsat = jnp.sum(assign < 0, axis=1, where=mask) > 1
        return unsat

    # unsat = {
    #     "xor": jnp.sum(unsats_xor(x0, xor.clauses.lits, xor.clauses.sign, xor.clauses.mask)),
    #     "cnf": jnp.sum(unsats_cnf(x0, cnf.clauses.lits, cnf.clauses.sign, cnf.clauses.mask)),
    #     "eo": jnp.sum(unsats_eo(x0, eo.clauses.lits, eo.clauses.sign, eo.clauses.mask)),
    #     "nae": jnp.sum(unsats_nae(x0, nae.clauses.lits, nae.clauses.sign, nae.clauses.mask)),
    #     "card": jnp.sum(unsats_card(x0, card.clauses.lits, card.clauses.sign, card.clauses.mask, card.cards)),
    #     "amo": jnp.sum(unsats_amo(x0, amo.clauses.lits, amo.clauses.sign, amo.clauses.mask)),
    # }
    # unsat = (
    #     (jnp.sum(unsats_xor(x0, xor.clauses.lits, xor.clauses.sign, xor.clauses.mask)))
    #     + (jnp.sum(unsats_cnf(x0, cnf.clauses.lits, cnf.clauses.sign, cnf.clauses.mask)))
    #     + (jnp.sum(unsats_eo(x0, eo.clauses.lits, eo.clauses.sign, eo.clauses.mask)))
    #     + (jnp.sum(unsats_nae(x0, nae.clauses.lits, nae.clauses.sign, nae.clauses.mask)))
    #     + (jnp.sum(unsats_card(x0, card.clauses.lits, card.clauses.sign, card.clauses.mask, card.cards)))
    #     + (jnp.sum(unsats_amo(x0, amo.clauses.lits, amo.clauses.sign, amo.clauses.mask)))
    # )
    unsat = jnp.array(
        [
            (jnp.sum(unsats_xor(x0, xor.clauses.lits, xor.clauses.sign, xor.clauses.mask))),
            (jnp.sum(unsats_cnf(x0, cnf.clauses.lits, cnf.clauses.sign, cnf.clauses.mask))),
            (jnp.sum(unsats_eo(x0, eo.clauses.lits, eo.clauses.sign, eo.clauses.mask))),
            (jnp.sum(unsats_nae(x0, nae.clauses.lits, nae.clauses.sign, nae.clauses.mask))),
            (jnp.sum(unsats_card(x0, card.clauses.lits, card.clauses.sign, card.clauses.mask, card.cards))),
            (jnp.sum(unsats_amo(x0, amo.clauses.lits, amo.clauses.sign, amo.clauses.mask))),
        ]
    )
    return unsat


def get_jax_object_size(obj: Iterable | Array):
    if isinstance(obj, jnp.ndarray) or isinstance(obj, Array):
        print(obj.nbytes / 1e9, "gb")
        return obj.nbytes
    elif isinstance(obj, Iterable):  # If it's a tuple/list, sum over elements
        return sum(get_jax_object_size(item) for item in obj)
    else:
        return 0  # Ignore non-JAX objects


def moreau_point_sampler(tasks: int, n_vars: int, fixed_vars: dict[int, int] = None) -> Array:
    # block cipher over index?
    # adaptive based on previous x0?
    # random uniform? snapped?
    # metropolis-hasting? simanneal? mcmc?
    pass


def hj_moreau_prox(x: Array, hyperparams: Any | None = None, scaling: float = 1):
    pass


def run_solver(
    tasks: int, n_vars: int, n_clause: int, batch: int, objs: tuple[Objective, ...], vals: Validators
) -> float:
    pg = ProjectedGradient(fun=fval, projection=projection_box, maxiter=50000)
    #prox = ProximalGradient(fun=fval, prox=hj_moreau_prox, maxiter=50000)

    def opt(x0: Array, objs: tuple[Objective, ...] = None, vals: Validators = None, weight: Array = None):
        # jax.debug.print("{}", x0)
        x0, state = pg.run(x0, hyperparams_proj=(-1, 1), objs=objs, weight=weight)
        # jax.debug.print("{}, {}", state, fval(x0, objs, weight))
        res = verify(x0, vals.xor, vals.cnf, vals.eo, vals.nae, vals.card, vals.amo)
        return x0, res, state.iter_num

    weight = 1  # jnp.ones(n_clause) # eventually objs.weight???
    v_opt = jax.jit(jax.vmap(opt, in_axes=(0, None, None, None), axis_name="batch"))
    p_opt = jax.pmap(v_opt, in_axes=(0, None, None, None))

    # Upward adjust #tasks for batch and gpu count.
    n_gpu = len(jax.devices("gpu"))
    tasks = ((tasks + (batch * n_gpu - 1)) // (batch * n_gpu)) * (batch * n_gpu)

    best_unsat = np.inf
    best_x = None
    tic = 0
    t0 = time()
    while time() - t0 < 300:
        key = jax.random.PRNGKey(tic)
        x0 = jax.random.truncated_normal(key, -1.0, 1.0, shape=(tasks, n_vars))
        xInit = x0[:]

        # x0 = jnp.sign(xS) # start at vertices
        # x0 = jnp.ones((1,n_vars)) # start at all false (1 job)
        # x0 = jnp.zeros((1,n_vars)) # start at all all zeros (1 job)

        # Random ramsey construction point (e.g. a valid assignment with 0 otherwise)
        # locs = jnp.where(jnp.arange(n_vars, dtype=float) % 3 == 0, -1, 0).reshape(1, n_vars)
        # x0 = jnp.ones((1,n_vars))
        # x0 = x0*locs

        res_x0 = []
        res_unsat = []
        res_iters = []
        cumul_best = jnp.inf

        pbar = tqdm(range(0, tasks // batch, n_gpu), "batches (best=inf)")
        for i in pbar:
            device_batches = []
            for d in range(n_gpu):
                start_idx = (i + d) * batch
                end_idx = start_idx + batch
                device_batches.append(x0[start_idx:end_idx])
            stacked_batches = jnp.stack(device_batches)

            # stamp1 = time()
            batch_x0, batch_unsat, batch_iters = p_opt(stacked_batches, objs, vals, weight)
            # stamp2 = time()
            # print("returned to python after", stamp2 - stamp1)

            # Collect results
            for d in range(n_gpu):
                res_x0.append(batch_x0[d])
                res_unsat.append(batch_unsat[d])
                res_iters.append(batch_iters[d])

            batch_best = jnp.min(jnp.sum(batch_unsat, axis=1))
            cumul_best = batch_best if batch_best < cumul_best else cumul_best
            if batch_best == 0:
                print(batch_unsat)
                print("count a cost 0????")
                break
            pbar.set_description("batches (best={})".format(cumul_best))

        x0 = jnp.concat(res_x0)[:tasks]
        unsat = jnp.concat(res_unsat)[:tasks]
        iters = jnp.concat(res_iters)[:tasks]

        print(f"PGD Iters: min: {jnp.min(iters)}, max: {jnp.max(iters)}, mean: {jnp.mean(iters)}")
        unsat = unsat.sum(axis=1)
        min_unsat = unsat.min()
        print("Restarts:", tic, "Best found unsat count:", min_unsat)

        # This is essentially the number of jobs that got the clause wrong.
        # Weight for a clause goes to zero if all got it right - contributes nothing to grad?
        # Blow up the grad for a clause that many got wrong
        # but starting at 1... then 0.9 + 0.1 * 1 = 0.1 = 1.... and eveything else is scaled back
        # reward = unsat.sum(axis=0)
        # weight = 0.9 * weight + 0.1 * reward / reward.max()
        # unsat = weight * unsat

        if min_unsat < best_unsat:
            best_unsat = min_unsat
            loc = jnp.argmin(unsat)
            best_x = x0[loc]
            best_x_start = xInit[loc]
            best_iters = iters[loc]

        stamp = time()
        print("Total iter time: {:.2f}/300.00 o {} | iters {}".format(stamp - t0, best_unsat, best_iters))

        if best_unsat == 0:
            print("SAT! at index {} with starting point {} (ending {})".format(loc, best_x_start, best_x))
            out_string = "v"
            assignment = []
            for i in range(n_vars):
                lit = i + 1
                if best_x[i] > 0:
                    out_string += " {}".format(-lit)
                    assignment.append(-lit)
                else:
                    out_string += " {}".format(lit)
                    assignment.append(lit)
            print(out_string)
            break
        tic += 1
    return stamp - t0


def main(dimacs: str = None, tasks: int = 32, batch: int = 16, mode: Opt[int] = None) -> None:
    if dimacs is None:
        print("Error: Please provide a (hybrid) dimacs CNF file")
        return 1

    sat = Formula()
    stamp1 = time()
    sat.read_DIMACS(dimacs)
    stamp2 = time()
    read_time = stamp2 - stamp1

    n_vars = sat.n_var
    n_clause = sat.n_clause
    objectives, validation = preprocess_all(sat, mode)
    stamp1 = time()
    process_time = stamp1 - stamp2

    del sat
    print("Running Solver")
    t_solve = run_solver(tasks, n_vars, n_clause, batch, objectives, validation)
    print("Some stats")
    print("Time reading input:", read_time)
    print("Time processing to Arrays:", process_time)
    # print("upper bound gpu bytes:", total_mem)
    print("Time spent solving", t_solve)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file with optional parameters")
    parser.add_argument("file", help="The file to process")
    parser.add_argument("-p", "--profile", action="store_true", help="Enable profiling")
    parser.add_argument("-t", "--tasks", type=int, default=32, help="Number of tasks to use")
    parser.add_argument("-b", "--batch", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("-m", "--mode", type=int, default=1, help="Which mode to use. Provide 0 to be prompted")

    args = parser.parse_args()

    # if "XLA_FLAGS" in os.environ:
    #     print(os.environ["XLA_FLAGS"])
    # else:
    #     print("CHECK XLA FLAGS")
    # Run with or without profiler based on the flag
    if args.profile:
        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
            main(args.file, args.tasks, args.batch, args.mode)
    else:
        main(args.file, args.tasks, args.batch, args.mode)
