import os
import sys
import typer
import argparse
from time import perf_counter as time

from concurrent.futures import ThreadPoolExecutor
import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box
from typing import Optional as Opt, NamedTuple
from jax.typing import ArrayLike as Array

from boolean_whf import ClauseGroup, ApproxLenClauses, class_map, Objective, ClauseArrays
from sat_loader import Formula

jax.config.update("jax_enable_x64", True)
from functools import partial
from jax.profiler import annotate_function, TraceAnnotation


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
    if mode is not None:
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
    objectives: list[Objective] = []
    validation: dict[str, Objective] = {}

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
    lits, sign, _, _ = objective.clauses
    dft, idft = objective.ffts
    forward_mask = objective.forward_mask

    # Evaluate ESPs
    assignment = sign * x[lits]
    prod = jnp.prod(dft + assignment[:, None, :], axis=2, where=forward_mask)
    clause_eval = jnp.sum(idft * prod, axis=1).real
    cost = weight * clause_eval
    return jnp.sum(cost)


@jax.jit
def fval_one_instrumented_sparse(x: Array, objective: Objective, weight: Array) -> Array:
    with TraceAnnotation("extract_clauses"):
        _, sign, _, onehot = objective.clauses
        dft, idft = objective.ffts
        forward_mask = objective.forward_mask

    with TraceAnnotation("compute_assignment"):
        # Use matrix multiplication instead of indexing
        xlits = jnp.einsum("v,clv->cl", x, onehot)
        assignment = sign * xlits  # x[lits]

    with TraceAnnotation("compute_prod"):
        prod = jnp.prod(dft + assignment[:, None, :], axis=2, where=forward_mask)

    with TraceAnnotation("compute_clause_eval"):
        clause_eval = jnp.sum(idft * prod, axis=1).real

    with TraceAnnotation("compute_cost"):
        cost = weight * clause_eval
        result = jnp.sum(cost)

    return result


@jax.jit
def fval(x: Array, objs: tuple[Objective], num_objs: int, weight: Array) -> Array:
    # TODO: Should the loop be a separate function which for jax.jit?
    # TODO: This might be the choke point.
    # TODO: JAX profiling.
    # TODO: Fix weight - keeping at 1 for now. #weight[i] eventually

    def fval_body(cumul_res, objective):
        jax.debug.print("{y}", y=objective[0])
        res = cumul_res + fval_one(x, objective[0], weight)
        return res, res

    # weight = 1
    # result = jax.lax.scan(fval_body, 0.0, objs)
    # jax.lax.fori_loop(0, num_objs, fval_body, 0.0)
    # return result
    if len(objs) == 1:
        cost = fval_one_instrumented_sparse(x, objs[0], weight)
        return cost
    else:
        result = jax.lax.fori_loop(0, num_objs, fval_body, 0.0)
        return result
    # for objective in objs:
    #     group_costs.append(fval_one(x, objective, weight))
    # return jnp.sum(jnp.array(group_costs))


@jax.jit
@partial(jax.profiler.annotate_function, name="verify_sparse")
def verify_sparse(
    x0: Array, xor: Objective, cnf: Objective, eo: Objective, nae: Objective, card: Objective, amo: Objective
) -> dict[str, Array] | Array:
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
@partial(jax.profiler.annotate_function, name="verify")
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
        unsat = jnp.sum(assign < 0, axis=1, where=mask) < cards
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


def run_solver(tasks: int, num_vars: int, num_clause: int, objs: list[Objective], vals: Validators) -> float:
    pg = ProjectedGradient(fun=fval, projection=projection_box, maxiter=500)

    def opt(x0: Array, objs: list[Objective], num_objs: int, vals: Validators, weight: Array):
        pg_res = pg.run(x0, hyperparams_proj=(-1, 1), objs=objs, num_objs=num_objs, weight=weight)
        x0 = pg_res.params
        iters = pg_res.state.iter_num
        res = verify(x0, vals.xor, vals.cnf, vals.eo, vals.nae, vals.card, vals.amo)
        return x0, res, iters

    def only_opt(x0: Array, objs: list[Objective], num_objs: int, weight: Array):
        pg_res = pg.run(x0, hyperparams_proj=(-1, 1), objs=objs, num_objs=num_objs, weight=weight)
        x0 = pg_res.params
        return x0

    def only_val(x0: Array, vals: Validators):
        res = verify_sparse(x0, vals.xor, vals.cnf, vals.eo, vals.nae, vals.card, vals.amo)
        return res

    p_opt = jax.jit(jax.vmap(opt, in_axes=(0, None, None, None, None)))
    vmap_optimize = jax.vmap(only_opt, in_axes=(0, None, None, None))
    vmap_verify = jax.vmap(only_val, in_axes=(0, None))

    num_objs = len(objs)
    best_unsat = np.inf
    best_x = None

    tic = 0
    t0 = time()
    while time() - t0 < 300:
        key = jax.random.PRNGKey(tic)
        # x0 = jax.random.uniform(key, (tasks, num_vars), minval=-1.0, maxval=1.0)
        x0 = jax.random.truncated_normal(key, -1.0, 1.0, shape=(tasks, num_vars))
        # TODO fix weight
        weight = jnp.ones(num_clause)

        # # Time just the optimization
        # start = time()
        # optimized_x0 = vmap_optimize(x0, objs, num_objs, weight)
        # optimize_time = time() - start
        # # Time just the verification
        # start = time()
        # unsat = vmap_verify(optimized_x0, vals)
        # verify_time = time() - start

        # print(f"Optimization time: {optimize_time}s")
        # print(f"Verification time: {verify_time}s")
        # stamp = time()
        # break

        # Optimization
        print("commencing projected gradient")
        stamp1 = time()
        x0, unsat, iters = p_opt(x0, objs, num_objs, vals, weight)
        stamp2 = time()
        print("returned to python after", stamp2 - stamp1)
        print(f"Min iters: {jnp.min(iters)}, Max iters: {jnp.max(iters)}, Mean: {jnp.mean(iters)}")

        unsat = unsat.sum(axis=1)
        min_unsat = unsat.min()
        print("loops:", tic, "error/unsat", min_unsat)

        # This is essentially the number of jobs that got the clause wrong.
        # Weight for a clause goes to zero if all got it right - contributes nothing to grad?
        # Blow up the grad for a clause that many got wrong
        # but starting at 1... then 0.9 + 0.1 * 1 = 0.1 = 1.... and eveything else is scaled back
        # reward = unsat.sum(axis=0)
        # weight = 0.9 * weight + 0.1 * reward / reward.max()
        # unsat = weight * unsat

        stamp = time()
        if min_unsat < best_unsat:
            best_unsat = min_unsat
            best_x = x0[jnp.argmin(unsat)]
            print("{:.2f}/300.00 o {}".format(stamp - t0, best_unsat))

        if best_unsat == 0:
            out_string = "v"
            assignment = []
            for i in range(num_vars):
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


def main(dimacs: str = None, threads: int = 32, mode: Opt[int] = None):
    if dimacs is None:
        print("Error: Please provide a (hybrid) dimacs CNF file")
        return 1

    sat = Formula()
    stamp1 = time()
    sat.read_DIMACS(dimacs)
    stamp2 = time()
    read_time = stamp2 - stamp1

    num_vars = sat.n_var
    num_clause = sat.n_clause
    objectives, validation = preprocess_all(sat, mode)
    stamp1 = time()
    process_time = stamp1 - stamp2

    total_mem = 0
    for ob in objectives:
        for item in ob:
            if isinstance(item, jax.Array):
                total_mem += sys.getsizeof(item)
                continue
            elif item is not None:
                for arr in item:
                    total_mem += sys.getsizeof(arr)
            continue

    del sat
    print("Running Solver")
    t_solve = run_solver(threads, num_vars, num_clause, objectives, validation)
    print("Some stats")
    print("Time reading input:", read_time)
    print("Time processing to Arrays:", process_time)
    print("upper bound gpu bytes:", total_mem)
    print("Time spent solving", t_solve)


if __name__ == "__main__":
    print(sys.version)
    parser = argparse.ArgumentParser(description="Process a file with optional parameters")
    parser.add_argument("file", help="The file to process")
    parser.add_argument("-p", "--profile", action="store_true", help="Enable profiling")
    parser.add_argument("-t", "--threads", type=int, default=32, help="Number of threads to use")
    parser.add_argument("-m", "--mode", type=int, default=None, help="Which mode to use (bypass prompt)")

    args = parser.parse_args()

    if "XLA_FLAGS" in os.environ:
        print(os.environ["XLA_FLAGS"])
    else:
        print("CHECK XLA FLAGS")
    # Run with or without profiler based on the flag
    if args.profile:
        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
            main(args.file, args.threads, args.mode)
    else:
        main(args.file, args.threads, args.mode)
