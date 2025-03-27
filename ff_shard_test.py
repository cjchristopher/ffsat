import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable pre-allocation
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform-specific allocator

import typer
import argparse
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter as time
from tqdm import tqdm
from typing import Optional as Opt, NamedTuple
from jax.typing import ArrayLike as Array

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_box

from boolean_whf import ClauseGroup, ApproxLenClauses, class_map, Objective, ClauseArrays
from sat_loader import Formula

jax.config.update("jax_enable_x64", True)


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
def verify(
    x0: Array, xor: Objective, cnf: Objective, eo: Objective, nae: Objective, card: Objective, amo: Objective
) -> Array:
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
        sat_count = jnp.sum(assign < 0, axis=1, where=mask)
        unsat = jnp.where(cards < 0, sat_count >= jnp.abs(cards), sat_count < cards)
        return unsat

    @jax.jit
    def unsats_amo(x0: Array, lits: Array, sign: Array, mask: Array):
        assign = sign * x0[lits]
        unsat = jnp.sum(assign < 0, axis=1, where=mask) > 1
        return unsat

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


def run_solver(
    tasks: int, n_vars: int, n_clause: int, batch: int, objs: tuple[Objective, ...], vals: Validators, debug_halt: int
) -> float:
    # Create device mesh
    devices = jax.devices("gpu")
    n_gpu = len(devices)
    mesh = Mesh(devices, ("device",))

    # Define partition specs for sharding
    obj_spec = P("device")  # Shard first dimension of arrays in objective
    batch_spec = P("device")  # Shard batch dimension across devices

    # Create a function that evaluates a single objective with internal sharding
    def get_sharded_evaluator(obj: Objective, weight: Array):
        """Create an evaluator for a single objective that handles internal sharding."""
        # Extract arrays from objective for explicit sharding
        lits, sign, _, _, _ = obj.clauses
        dft, idft = obj.ffts
        forward_mask = obj.forward_mask

        # Create a sharding specification for clause dimension
        clause_sharding = NamedSharding(mesh, obj_spec)

        # Create evaluation function that uses sharded arrays
        def evaluate(x):
            # Apply explicit sharding to input arrays before computation
            # This is key - we tell JAX to shard these arrays explicitly at the start
            with mesh:
                # Add annotations to source arrays
                lits_sharded = jax.device_put(lits, clause_sharding)
                sign_sharded = jax.device_put(sign, clause_sharding)
                dft_sharded = jax.device_put(dft, clause_sharding)
                idft_sharded = jax.device_put(idft, clause_sharding)
                mask_sharded = jax.device_put(forward_mask, clause_sharding)

                # Compute with sharded arrays
                assignment = sign_sharded * x[lits_sharded]
                prod = jnp.prod(dft_sharded + assignment[:, None, :], axis=2, where=mask_sharded)
                clause_eval = jnp.sum(idft_sharded * prod, axis=1).real
                cost = weight * clause_eval
                return jnp.sum(cost)

        return evaluate

    # Create a function that evaluates all objectives for a given input
    def get_full_evaluator(objs: tuple[Objective, ...], weight: Array):
        """Create an evaluator for all objectives with proper sharding."""
        # Create evaluators for each objective
        evaluators = [get_sharded_evaluator(obj, weight) for obj in objs]

        # Combine evaluators
        def evaluate_all(x):
            costs = [eval_fn(x) for eval_fn in evaluators]
            return jnp.sum(jnp.array(costs))

        return evaluate_all

    # Upward adjust #tasks for batch and gpu count
    batch_size = batch * n_gpu
    tasks = ((tasks + batch_size - 1) // (batch_size)) * (batch_size)

    weight = jnp.ones(1)
    best_unsat = np.inf
    best_x = None
    tic = 0
    t0 = time()

    while time() - t0 < 300:
        key = jax.random.PRNGKey(tic)
        x0 = jax.random.truncated_normal(key, -1.0, 1.0, shape=(tasks, n_vars))
        xInit = x0[:]
        res_x0 = []
        res_unsat = []
        res_iters = []
        cumul_best = jnp.inf

        # Create a jitted function for optimizing a batch of inputs
        def process_batch(x_batch: Array, weight: Array):
            """Process a batch of inputs with sharding."""
            # Create evaluator and optimizer
            full_evaluator = get_full_evaluator(objs, weight)
            pg = ProjectedGradient(fun=full_evaluator, projection=projection_box, maxiter=50000)

            # Use vmap to process multiple inputs in parallel
            def optimize_one(x0):
                # Run optimization
                x_opt, state = pg.run(x0, hyperparams_proj=(-1, 1))
                # Verify solution
                res = verify(x_opt, vals.xor, vals.cnf, vals.eo, vals.nae, vals.card, vals.amo)
                return x_opt, res, state.iter_num

            # Use vmap with proper in_axes (we're handling device distribution separately)
            optimize_batch = jax.vmap(optimize_one)

            # Apply the batch optimization with explicit device placement
            x0_sharded = jax.lax.with_sharding_constraint(x_batch, NamedSharding(mesh, batch_spec))
            results = optimize_batch(x0_sharded)
            return results

        jit_batch = jax.jit(process_batch, in_shardings=(NamedSharding(mesh, batch_spec), None))

        if debug_halt:
            dummy_batch = jnp.ones((batch_size, n_vars))
            dummy_weight = jnp.ones(1)
            try:
                jaxpr = jax.make_jaxpr(process_batch)(dummy_batch, dummy_weight)
                print(jaxpr)
            except Exception as e:
                print(f"Error making jaxpr: {e}")
            # Stop execution if debug_halt is set
            print("\nDebug halt requested, exiting")
            return 0

        # Process in batches suitable for devices
        pbar = tqdm(range(0, tasks, batch_size), "batches (best=inf)")
        for i in pbar:
            # Get batch
            end_idx = min(i + batch_size, tasks)
            batch_x0 = x0[i:end_idx]

            # Pad if needed (to avoid JAX shape errors)
            if batch_x0.shape[0] < batch_size:
                # Use numpy to handle the condition
                pad_size = int(batch_size - batch_x0.shape[0])
                batch_x0 = jnp.pad(batch_x0, ((0, pad_size), (0, 0)))

            # Run optimization
            opt_x0, opt_unsat, opt_iters = jit_batch(batch_x0, weight)

            # Store only valid results (no padding)
            valid_size = min(batch_size, tasks - i)
            res_x0.append(opt_x0[:valid_size])
            res_unsat.append(opt_unsat[:valid_size])
            res_iters.append(opt_iters[:valid_size])

            # Check for solution
            # print(opt_unsat, opt_unsat.shape)
            batch_best = jnp.min(jnp.sum(opt_unsat, axis=1))
            cumul_best = batch_best if batch_best < cumul_best else cumul_best
            if batch_best == 0:
                print(opt_unsat)
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
            jax.numpy.save("start_x.npy", best_x_start)
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


def main(dimacs: str = None, tasks: int = 32, batch: int = 16, mode: Opt[int] = None, init_x: str = None, debug_halt: int = 0) -> None:
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
    t_solve = run_solver(tasks, n_vars, n_clause, batch, objectives, validation, init_x, debug_halt)
    print("Some stats")
    print("Time reading input:", read_time)
    print("Time processing to Arrays:", process_time)
    print("Time spent solving:", t_solve)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file with optional parameters")
    parser.add_argument("file", help="The file to process")
    parser.add_argument("-p", "--profile", action="store_true", help="Enable profiling")
    parser.add_argument("-t", "--tasks", type=int, default=32, help="Number of tasks to use")
    parser.add_argument("-b", "--batch", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("-m", "--mode", type=int, default=1, help="Which mode to use. Provide 0 to be prompted")
    parser.add_argument("-i", "--init_x", type=str, default=None, help="An initial array of x starting points in .npy format. Overrides -t and -b.")
    parser.add_argument(
        "-d",
        "--debug_jaxpr",
        type=int,
        default=0,
        help="Will emit the jax expression of the JIT search and exit. It is recommended you pipe output to file.",
    )

    args = parser.parse_args()

    # Run with or without profiler based on the flag
    if args.profile:
        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
            main(args.file, args.tasks, args.batch, args.mode, args.init_x, args.debug_jaxpr)
    else:
        main(args.file, args.tasks, args.batch, args.mode, args.init_x, args.debug_jaxpr)
