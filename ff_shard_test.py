import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable pre-allocation
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform-specific allocator
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=true --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_highest_priority_async_stream=true"
)

from argparse import ArgumentParser as ArgParse
from collections.abc import Callable
from time import perf_counter as time
from typing import Optional as Opt

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike as Array
from jaxopt import LBFGSB, ProjectedGradient, ProximalGradient
from jaxopt.projection import projection_box
from tqdm import tqdm

from boolean_whf import Objective, class_idno
from sat_loader import Formula
from utils import preprocess_to_matrix

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_checks", True)
# jax.config.update("jax_disable_jit", True)


def shard_objective(obj: Objective, clause_sharding: NamedSharding) -> Objective:
    # Shard the objcetive arrays along the first (clauses) dimension since eval is row-wise.
    def shard_leaf(leaf):
        if isinstance(leaf, jax.Array):
            return jax.device_put(leaf, clause_sharding)
        return leaf

    return jax.tree_util.tree_map(shard_leaf, obj)


def eval_verify(objs: tuple[Objective, ...], combined: bool) -> tuple[Callable, Callable]:
    """
    Constructs JAX-based evaluators and verifiers for a set of objectives.
    This function generates callable evaluators and verifiers for the given objectives.
    Evaluators compute the cost of assignments, while verifiers check the satisfaction
    of clauses based on the assignment. The function supports both single and batch
    processing modes.
    Args:
        objs (tuple[Objective, ...]): A tuple of Objective instances, each containing
            clauses, FFTs, and other necessary data for evaluation and verification.
        combined (bool): If True, returns batch processing functions for evaluation
            and verification. If False, returns single-instance processing functions.
    Returns:
        tuple: A tuple containing two callables:
            - If `combined` is False:
                - evaluate_all (Callable[[Array, list[Array]], tuple[Array, Array]]):
                  Evaluates the cost for all objectives given a single assignment and weights.
                - verify_all (Callable[[Array], Array]): Verifies the satisfaction of all
                  objectives for a single assignment.
            - If `combined` is True:
                - evaluate_batch (Callable[[Array, list[Array]], Array]): Evaluates the cost
                  for all objectives in a batch of assignments and weights.
                - verify_batch (Callable[[Array], Array]): Verifies the satisfaction of all
                  objectives for a batch of assignments.
    """

    # Construct JAX sharded evaluator and verifier
    def single_eval_verify(obj: Objective) -> tuple[Callable[[Array], Array], ...]:
        lits = obj.clauses.lits
        sign = obj.clauses.sign
        mask = obj.clauses.mask
        cards = obj.clauses.cards
        types = obj.clauses.types
        # sparse = obj.clauses.sparse

        dft, idft = obj.ffts
        forward_mask = obj.forward_mask

        def evaluate(x: Array, weight: Array) -> Array:
            assignment = sign * x[lits]
            prod = jnp.prod(dft + assignment[:, None, :], axis=2, where=forward_mask)
            clause_eval = jnp.sum(idft * prod, axis=1).real
            cost = weight * clause_eval
            return jnp.sum(cost)

        def verify(x: Array) -> Array:
            # N.B. This logic of the output is inverted w.r.t inution.
            # E.g. We check if the clause is UNSAT (e.g. True in the vector => that clause is UNSAT)
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

    def evaluate_all(x: Array, weights: list[Array]) -> tuple[Array, Array]:
        costs = [eval_fn(x, weight) for (eval_fn, weight) in zip(evaluators, weights)]
        cost = jnp.sum(jnp.array(costs))
        return cost, cost  # aux info

    def evaluate_batch(xB: Array, weights: list[Array]) -> Array:
        # def eval_single(x):
        #     return evaluate_all(x, weights)
        # batch_eval, _ = jax.lax.map(eval_single, xB)
        batch_eval, _ = jax.vmap(evaluate_all, in_axes=(0, None))(xB, weights)
        return jnp.mean(batch_eval), batch_eval

    def verify_all(x: Array) -> Array:
        all_res = [res_fn(x) for res_fn in verifiers]
        res = jnp.concat(all_res)
        return res

    def verify_batch(xB: Array) -> Array:
        batch_res = jax.vmap(verify_all)(xB)
        return batch_res

    if not combined:
        return evaluate_all, verify_all

    return evaluate_batch, verify_batch


def run_solver(
    timeout: int,
    n_vars: int,
    batch_sz: int,
    restart: int,
    fuzz_limit: int,
    opt_combined: bool,
    objs: tuple[Objective, ...],
    v_start: bool = False,
) -> float:
    # Create device mesh, declare JAX specs, and shard:
    devices = jax.devices("gpu")
    gpu_batch = batch_sz * len(devices)
    weights = [jnp.ones_like(obj.clauses.types) for obj in objs]
    mesh = Mesh(devices, ("device",))
    shard_spec = P("device")
    restart = np.inf if not restart else restart

    best_unsat = jnp.inf
    best_x = None
    best_start = None

    # all_x0 = []
    # all_unsat = []
    all_iters = []
    all_unsat_cts = []
    starts = 0
    restart_ct = 0
    key = jax.random.PRNGKey(restart_ct)
    fuzz_key = jax.random.PRNGKey(restart_ct + 1)
    t0 = time()

    with mesh:
        clause_sharding = NamedSharding(mesh, shard_spec)
        sharded_objs = tuple(shard_objective(obj, clause_sharding) for obj in objs)
        sharded_weight = [jax.device_put(weight, clause_sharding) for weight in weights]

        def process_batch(x_batch: Array, weights: list[Array]) -> tuple[Array, Array, Array, Array]:
            evaluator, verifier = eval_verify(sharded_objs, opt_combined)

            pg = ProjectedGradient(fun=evaluator, projection=projection_box, maxiter=50, has_aux=True)
            # bfgs = LBFGSB(fun=evaluator, maxiter=50, has_aux=True)

            def optimize(x: Array) -> tuple[Array, Array, Array]:
                x_opt, state = pg.run(x, hyperparams_proj=(-1, 1), weights=weights)
                # x_opt, state = bfgs.run(x, bounds=(-1 * jnp.ones_like(x), jnp.ones_like(x)), weights=weights)
                unsat = jnp.squeeze(verifier(x_opt))
                return x_opt, unsat, jnp.atleast_1d(state.iter_num)

            if not opt_combined:
                optimize_batch = jax.vmap(optimize)
                x_shard = jax.lax.with_sharding_constraint(x_batch, clause_sharding)
            else:
                optimize_batch = jax.jit(optimize)
                x_shard = x_batch
            x_opt, unsat, iters = optimize_batch(x_shard)
            unsat_cl_count = jnp.sum(unsat, axis=0)
            return x_opt, jnp.sum(unsat, axis=1), iters, unsat_cl_count

        if not opt_combined:
            opt_batch = jax.jit(process_batch, in_shardings=(clause_sharding, [clause_sharding] * len(weights)))
        else:
            opt_batch = jax.jit(process_batch, in_shardings=(None, [clause_sharding] * len(weights)))

        batch_time = t0
        pbar = tqdm(
            total=timeout,
            desc=f"restart {restart_ct} ({starts}/{restart} -- best={best_unsat})",
            bar_format="{l_bar}{bar}| {elapsed}s/{total}s",
        )
        while time() - t0 < timeout:
            x0 = jax.random.uniform(key, minval=-1.0, maxval=1.0, shape=(gpu_batch, n_vars))
            fuzz = jax.random.uniform(fuzz_key, minval=1e-10, maxval=1e-4, shape=x0.shape)
            fuzz_attempt = 0

            if v_start:
                # snap to nearest vertex and fuzz to avoid immediate saddle
                x0 = jnp.sign(x0)
                x0 = x0 + -x0 * fuzz
            x_starts = x0[:]

            # Do it.
            opt_x0, opt_unsat, opt_iters, opt_unsat_ct = opt_batch(x0, sharded_weight)

            batch_best = jnp.min(opt_unsat)
            best_unsat = batch_best if batch_best < best_unsat else best_unsat

            found_sol = False
            if batch_best == 0:
                print("Found a solution!")
                found_sol = True
            elif fuzz_limit:
                # Solution not found - fuzz current convergence.
                while fuzz_attempt < fuzz_limit:
                    fuzz_attempt += 1
                    xF = opt_x0 + -jnp.sign(opt_x0) * fuzz * 100
                    opt_xF, opt_unsatF, opt_itersF, opt_unsat_ctF = opt_batch(xF, sharded_weight)
                    batch_bestF = jnp.min(opt_unsatF)
                    if batch_bestF < best_unsat:
                        best_unsat = batch_bestF
                        opt_x0, opt_unsat, opt_iters, opt_unsat_ct = opt_xF, opt_unsatF, opt_itersF, opt_unsat_ctF
                    if batch_bestF == 0:
                        print(f"Fuzz {fuzz_attempt} found a solution!")
                        found_sol = True
                        break

            loc = jnp.argmin(opt_unsat)
            best_x = opt_x0[loc]
            # best_start = x_starts[loc]
            # best_iters = opt_iters[loc]

            if found_sol:
                # print("SAT! at index {} with starting point {} (ending {})".format(loc, best_start, best_x))
                print("SAT! at index {}".format(max((starts - 1), 0) + loc))
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
                stamp = time()
                break

            all_iters.append(opt_iters)
            all_unsat_cts.append(opt_unsat_ct)
            starts += gpu_batch
            if starts >= restart:
                # Statistics for this restart
                iters = jnp.concat(all_iters)
                unsat_ct = jnp.array(all_unsat_cts)

                # Reset/update for next restart
                restart_ct += 1
                key = jax.random.PRNGKey(restart_ct)
                fuzz_key = jax.random.PRNGKey(restart_ct + 1)
                starts = 0
                all_iters = []
                all_unsat_cts = []

                print(f"Optim Iters: min: {jnp.min(iters)}, max: {jnp.max(iters)}, mean: {jnp.mean(iters)}")
                print(f"Restarts: {restart_ct} | Current  (#unsat): {best_unsat}")

                punishment = unsat_ct.sum(axis=0)
                p_start = 0
                for idx, weight in enumerate(weights):
                    n_clause = len(weight)
                    p_end = p_start + n_clause
                    weights[idx] = 0.9 * weight + 0.1 * punishment[p_start:p_end] / punishment.max()
                    p_start += n_clause
                sharded_weight = [jax.device_put(weight, clause_sharding) for weight in weights]

                stamp = time()
                print("Restart time: {:.2f}/{}".format(stamp - t0, timeout))

            pbar.set_description(f"restart {restart_ct} ({starts}/{restart} -- best={best_unsat})")
            pbar.update(time() - batch_time)
            batch_time = time()
        pbar.close()
    return stamp - t0


def main(
    dimacs: str, mode: Opt[int], timeout: int, batch: int, restart: int, fuzz: int, combine: bool, vertex: bool
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
    # n_clause = sat.n_clause
    objectives, _ = preprocess_to_matrix(sat, mode)
    stamp1 = time()
    process_time = stamp1 - stamp2

    del sat
    print("Running Solver")
    t_solve = run_solver(timeout, n_vars, batch, restart, fuzz, combine, objectives, v_start=vertex)
    print("Some stats")
    print("Time reading input:", read_time)
    print("Time processing to Arrays:", process_time)
    print("Time spent solving:", t_solve)


if __name__ == "__main__":
    ap = ArgParse(description="Process a file with optional parameters")
    ap.add_argument("file", help="The file to process")
    ap.add_argument("-p", "--profile", action="store_true", help="Enable profiling")
    ap.add_argument("-t", "--timeout", type=int, default=300, help="Maximum time to run (timeout seconds)")
    ap.add_argument("-b", "--batch", type=int, default=16, help="Batch size per GPU")
    ap.add_argument("-f", "--fuzz", type=int, default=0, help="Number of times to attempt fuzzing per batch")
    ap.add_argument("-v", "--vertex", action="store_true", help="Start near a vertices")
    ap.add_argument(
        "-c", "--combine", action="store_true", help="Optimise a batch of points with one call to the optimiser"
    )
    ap.add_argument(
        "-r",
        "--restart",
        type=int,
        default=0,
        help="Points to test before adjusting weight and restarting (no weight/restart if 0)",
    )
    ap.add_argument(
        "-m",
        "--mode",
        type=int,
        default=0,
        help="Which clause partitioning mode to use. Mode 0 to be prompted after reading input",
    )

    args = ap.parse_args()

    # Run with or without profiler based on the flag
    print("Running with args:", args.mode, args.timeout, args.batch, args.restart, args.fuzz, args.combine)
    if args.profile:
        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
            main(args.file, args.mode, args.timeout, args.batch, args.restart, args.fuzz, args.combine, args.vertex)
    else:
        main(args.file, args.mode, args.timeout, args.batch, args.restart, args.fuzz, args.combine, args.vertex)
