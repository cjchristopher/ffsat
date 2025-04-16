import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable pre-allocation
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform-specific allocator

import argparse
from collections.abc import Callable
from time import perf_counter as time
from typing import Optional as Opt

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike as Array
from jaxopt import LBFGSB, ProjectedGradient, ScipyBoundedMinimize
from jaxopt.projection import projection_box
from tqdm import tqdm

from boolean_whf import Objective, class_idno
from sat_loader import Formula
from utils import preprocess_to_matrix

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_checks", True)
# jax.config.update("jax_disable_jit", True)


def shard_objective(obj: Objective, mesh: Mesh, shard_spec: P) -> Objective:
    # Objective sharding - shard the arrays along the first (clauses) dimension since eval is row-wise.
    def shard_leaf(leaf):
        if isinstance(leaf, jax.Array):
            clause_sharding = NamedSharding(mesh, shard_spec)
            return jax.device_put(leaf, clause_sharding)
        return leaf

    return jax.tree_util.tree_map(shard_leaf, obj)


def eval_verify(
    objs: tuple[Objective, ...],
) -> tuple[Callable[[Array], tuple[Array, Array]], Callable[[Array], Array]]:
    # Construct JAX sharded evaluator and verifier
    def single_eval_verify(obj: Objective) -> tuple[Callable[[Array], Array], ...]:
        lits = obj.clauses.lits
        sign = obj.clauses.sign
        weight = obj.clauses.weight
        mask = obj.clauses.mask
        cards = obj.clauses.cards
        types = obj.clauses.types
        # sparse = obj.clauses.sparse

        dft, idft = obj.ffts
        forward_mask = obj.forward_mask

        def evaluate(x: Array) -> Array:
            assignment = sign * x[lits]
            prod = jnp.prod(dft + assignment[:, None, :], axis=2, where=forward_mask)
            clause_eval = jnp.sum(idft * prod, axis=1).real
            cost = weight * clause_eval
            # if verbose:
            #    jax.debug.print("Score: {},{}", jnp.sum(cost), jnp.sum(cost).shape)
            return jnp.sum(cost)

        def verify(x: Array) -> Array:
            assign = sign * x[lits]
            # assign = sign * jnp.einsum("v,clv->cl", x0, sparse) #sparse version.

            unsat = jnp.zeros_like(types, dtype=bool)

            # xor (unsat if an even number of true (<0) assignments)
            unsat_cond = jnp.sum(assign < 0, axis=1, where=mask) % 2 == 0
            unsat_type = jnp.where(types == class_idno["xor"], unsat_cond, False)
            unsat = unsat | unsat_type

            # cnf (unsat if min value in assignment is false (>0))
            unsat_cond = jnp.min(assign, axis=1, where=mask, initial=jnp.inf) > 0
            unsat_type = jnp.where(types == class_idno["cnf"], unsat_cond, False)
            unsat = unsat | unsat_type

            # eo (unsat if true (<0) count != 1)
            unsat_cond = jnp.sum(assign < 0, axis=1, where=mask) != 1
            unsat_type = jnp.where(types == class_idno["eo"], unsat_cond, False)
            unsat = unsat | unsat_type

            # amo (unsat if true (<0) count >1)
            unsat_cond = jnp.sum(assign < 0, axis=1, where=mask) > 1
            unsat_type = jnp.where(types == class_idno["amo"], unsat_cond, False)
            unsat = unsat | unsat_type

            # nae (unsat IF NOT(AND(any_true, any_false))=T
            unsat_cond = jnp.logical_not(
                jnp.logical_and(
                    (jnp.min(assign, axis=1, where=mask, initial=jnp.inf) < 0),  # any true?
                    (jnp.max(assign, axis=1, where=mask, initial=-jnp.inf) > 0),  # any false?
                )
            )
            unsat_type = jnp.where(types == class_idno["nae"], unsat_cond, False)
            unsat = unsat | unsat_type

            # card
            card_count = jnp.sum(assign < 0, axis=1, where=mask)
            unsat_cond = jnp.where(cards < 0, card_count >= jnp.abs(cards), card_count < cards)
            unsat_type = jnp.where(types == class_idno["card"], unsat_cond, False)
            unsat = unsat | unsat_type
            return unsat

        return evaluate, verify

    evaluators, verifiers = zip(*[single_eval_verify(obj) for obj in objs])

    def evaluate_all(x: Array) -> tuple[Array, Array]:
        costs = [eval_fn(x) for eval_fn in evaluators]
        cost = jnp.sum(jnp.array(costs))
        return cost, cost  # aux info

    def verify_all(x: Array) -> Array:
        all_res = [res_fn(x) for res_fn in verifiers]
        res = jnp.array(all_res)
        return res

    return evaluate_all, verify_all


def next_x_batch(batch_x: Array) -> Array:
    pass


def run_solver(
    tasks: int, n_vars: int, n_clause: int, batch: int, objs: tuple[Objective, ...], init_x: Opt[str], debug_halt: int
) -> float:
    # Create device mesh, declare JAX specs, and shard:
    devices = jax.devices("gpu")
    n_gpu = len(devices)
    mesh = Mesh(devices, ("device",))
    shard_spec = P("device")

    with mesh:
        sharded_objs = tuple(shard_objective(obj, mesh, shard_spec) for obj in objs)

    # Upward adjust #tasks for batch and gpu count
    batch_size = batch * n_gpu
    tasks = ((tasks + batch_size - 1) // (batch_size)) * (batch_size)

    best_unsat = np.inf
    best_x = None
    tic = 0
    t0 = time()

    while time() - t0 < 300:
        key = jax.random.PRNGKey(tic)
        x0 = jax.random.truncated_normal(key, -1.0, 1.0, shape=(tasks, n_vars))
        if init_x:
            x0 = jnp.load(init_x)  # jax.numpy.save("start_x.npy", best_x_start)
            if len(x0.shape) == 1:
                x0 = x0.reshape(1, x0.shape[0])
            tasks = x0.shape[0]
            batch = 16
        xInit = x0[:]
        res_x0 = []
        res_unsat = []
        res_iters = []
        cumul_best = jnp.inf

        # Create a jitted function for optimizing a batch of inputs
        def process_batch(x_batch: Array):
            """Process a batch of inputs with sharding."""
            evaluator, verifier = eval_verify(sharded_objs)

            # pg = ProjectedGradient(fun=full_evaluator, projection=projection_box, maxiter=500, verbose=True, has_aux=True)
            pg = LBFGSB(fun=evaluator, maxiter=500, verbose=True, has_aux=True)

            def optimize_one(x0):
                # x_opt, state = pg.run(x0, hyperparams_proj=(-1, 1))
                # jax.debug.print("X: {}", x0)
                x_opt, state = pg.run(x0, bounds=(-1 * jnp.ones_like(x0), jnp.ones_like(x0)))
                # jax.debug.print("X: {}, \n {}", x_opt, x_opt == x0)
                res = verifier(x_opt)
                return x_opt, res, state.iter_num

            # Declare parallelisation for optimiser for a single starting point in a sharded batch of points, and run.
            optimize_batch = jax.vmap(optimize_one)
            x0_sharded = jax.lax.with_sharding_constraint(x_batch, NamedSharding(mesh, shard_spec))
            results = optimize_batch(x0_sharded)
            return results

        jit_batch = jax.jit(process_batch, in_shardings=(NamedSharding(mesh, shard_spec), None))

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
        batch_bar = tqdm(range(0, tasks, batch_size), "batches (best=inf)")
        for i in batch_bar:
            # Get batch
            end_idx = min(i + batch_size, tasks)
            batch_x0 = jnp.sign(x0[i:end_idx])

            # Pad if needed (to avoid JAX shape errors)
            if batch_x0.shape[0] < batch_size:
                print("WARNING: PADDING BATCH")
                # Use numpy to handle the condition
                pad_size = int(batch_size - batch_x0.shape[0])
                batch_x0 = jnp.pad(batch_x0, ((0, pad_size), (0, 0)))

            # Run optimization
            opt_x0, opt_unsat, opt_iters = jit_batch(batch_x0)

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
            batch_bar.set_description("batches (best={})".format(cumul_best))

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
            print(best_x)
            best_x_start = xInit[loc]
            best_iters = iters[loc]

        stamp = time()
        print("Total iter time: {:.2f}/300.00 o {} | iters {}".format(stamp - t0, best_unsat, best_iters))

        if best_unsat == 0:
            print("SAT! at index {} with starting point {} (ending {})".format(loc, best_x_start, best_x))
            # jax.numpy.save("start_x.npy", best_x_start)
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


def main(
    dimacs: str = None, tasks: int = 32, batch: int = 16, mode: Opt[int] = None, init_x: str = None, debug_halt: int = 0
) -> None:
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
    objectives, _ = preprocess_to_matrix(sat, mode)
    stamp1 = time()
    process_time = stamp1 - stamp2

    del sat
    print("Running Solver")
    t_solve = run_solver(tasks, n_vars, n_clause, batch, objectives, init_x, debug_halt)
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
    parser.add_argument(
        "-i",
        "--init_x",
        type=str,
        default=None,
        help="An initial array of x starting points in .npy format. Overrides -t and -b.",
    )
    parser.add_argument(
        "-d",
        "--debug_jaxpr",
        type=int,
        default=0,
        help="Will emit the jax expression of the JITted algorithm and exit. It is recommended you direect output to file.",
    )

    args = parser.parse_args()

    # Run with or without profiler based on the flag
    if args.profile:
        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
            main(args.file, args.tasks, args.batch, args.mode, args.init_x, args.debug_jaxpr)
    else:
        main(args.file, args.tasks, args.batch, args.mode, args.init_x, args.debug_jaxpr)
