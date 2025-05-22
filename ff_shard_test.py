from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable pre-allocation
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform-specific allocator
os.environ["XLA_FLAGS"] = " ".join(
    [
        "--xla_gpu_triton_gemm_any=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_highest_priority_async_stream=true",
        "--xla_gpu_enable_fast_min_max=true",
        "--xla_gpu_enable_cublaslt=true",
        "--xla_gpu_exhaustive_tiling_search=true",
    ]
)

# Single-host, multi-device computation on NVIDIA GPUs
# os.environ.update({
#   "NCCL_LL128_BUFFSIZE": "-2",
#   "NCCL_LL_BUFFSIZE": "-2",
#    "NCCL_PROTO": "SIMPLE,LL,LL128",
#  })

from argparse import ArgumentParser as ArgParse
from collections.abc import Callable
from contextlib import nullcontext
from time import perf_counter as time

import jax
import jax.numpy as jnp
#from jax.experimental import sparse as jsparse
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike as Array
from jaxopt import LBFGSB, ProjectedGradient, ProximalGradient
from jaxopt.projection import projection_box
from tqdm import tqdm

from boolean_whf import Objective, class_idno
from sat_loader import Formula
from utils import process_clauses

jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
#jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_optimization_level", "O3")
# jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_no_tracing", True)
# jax.config.update("jax_disable_jit", True)

# DEBUGGING BLOCK
# import logging
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_checks", True)
# jax.config.update("jax_explain_cache_misses", True)
# jax.config.update("jax_check_tracer_leaks", True)
# fh = logging.FileHandler(filename="jax.log")
# fh.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
# jaxlog = logging.getLogger("jax")
# # Remove any existing handlers
# for handler in jaxlog.handlers[:]:
#     jaxlog.removeHandler(handler)
# jaxlog.setLevel(logging.DEBUG)
# jaxlog.addHandler(fh)
# jaxlog.propagate = False
import functools

print = functools.partial(print, flush=True)

def nan_check(assignment, prod, clause_score, cost):
    if jnp.any(jnp.isnan(assignment)) or jnp.any(jnp.isnan(cost)) or jnp.any(jnp.isnan(prod)) or jnp.any(jnp.isnan(clause_score)):
        x = f"assignment: {assignment}, prod: {prod}, clause score: {clause_score}, weighted {cost}"
        print(x)

def nan_check_2(array, name):
    if jnp.any(jnp.isnan(array)):
        x = f"{name}: {array}"
        print(x)

def shard_objective(obj: Objective, clause_sharding: NamedSharding) -> Objective:
    # Shard the objcetive arrays along the first (clauses) dimension since eval is row-wise.
    def shard_leaf(leaf):
        if isinstance(leaf, jax.Array):
            return jax.device_put(leaf, clause_sharding)
        return leaf

    return jax.tree_util.tree_map(shard_leaf, obj)


def eval_verify(objs: tuple[Objective, ...]) -> tuple[Callable, Callable]:
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
        #sparse = obj.clauses.sparse

        dft, idft = obj.ffts
        forward_mask = obj.forward_mask

        @jax.jit
        def evaluate(x: Array, weight: Array) -> Array:
            # JAX don't appear to be continuing with sparse array support.
            #sparse_xlit = BCOO.fromdense(sparse) # nse = ? - figure this out.
            #xlits = jnp.einsum("v,clv->cl", x, sparse_xlit)
            #xlits = jnp.einsum("v,clv->cl", x, sparse)
            #assignment = sign * xlits
            assignment = sign * x[lits]
            prod = jnp.prod(dft + assignment[:, None, :], axis=2, where=forward_mask)
            clause_eval = jnp.sum(idft * prod, axis=1).real
            # TODO: think about this
            # cost = jnp.where(clause_eval > 0, clause_eval * weight, clause_eval)
            cost = weight * clause_eval
            #jax.lax.cond(jnp.any(jnp.isnan(prod)), lambda: jax.debug.callback(nan_check_2, assignment, "assignment") or jax.debug.callback(nan_check_2, prod, "prod"), lambda: None)
            #jax.debug.callback(nan_check, assignment, prod, clause_eval, cost, ordered=True, partitioned=True)
            return jnp.sum(cost)

        @jax.jit
        def verify(x: Array) -> Array:
            assign = sign * x[lits]
            unsat = jnp.zeros_like(types, dtype=bool)

            clause_unsat_rules = {
                "xor": lambda: jnp.sum(assign < 0, axis=1, where=mask) % 2 == 0,
                "cnf": lambda: jnp.min(assign, axis=1, where=mask, initial=jnp.inf) > 0,
                "eo": lambda: jnp.sum(assign < 0, axis=1, where=mask) != 1,
                "amo": lambda: jnp.sum(assign < 0, axis=1, where=mask) > 1,
                "nae": lambda: jnp.logical_not(
                    jnp.logical_and(
                        (jnp.min(assign, axis=1, where=mask, initial=jnp.inf) < 0),
                        (jnp.max(assign, axis=1, where=mask, initial=-jnp.inf) > 0),
                    )
                ),
                "card": lambda: jnp.where(
                    cards < 0,
                    jnp.sum(assign < 0, axis=1, where=mask) >= jnp.abs(cards),
                    jnp.sum(assign < 0, axis=1, where=mask) < cards,
                ),
            }

            for clause_type, handler in clause_unsat_rules.items():
                type_id = class_idno[clause_type]
                unsat_cond = handler()
                unsat = unsat | jnp.where(types == type_id, unsat_cond, unsat)

            return unsat

        return evaluate, verify

    evaluators, verifiers = zip(*[single_eval_verify(obj) for obj in objs])

    def evaluate_all(x: Array, weights: list[Array]) -> tuple[Array, Array]:
        costs = [eval_fn(x, weight) for (eval_fn, weight) in zip(evaluators, weights)]
        cost = jnp.sum(jnp.array(costs))
        return cost, cost  # aux info

    def verify_all(x: Array) -> Array:
        all_res = [res_fn(x) for res_fn in verifiers]
        res = jnp.concat(all_res)
        return res

    return evaluate_all, verify_all


def eval_verify_combined(objs: tuple[Objective, ...]) -> tuple[Callable, Callable]:
    evaluate_all, verify_all = eval_verify(objs)

    def evaluate_batch(x_batch: Array, weights: list[Array]) -> Array:
        batch_eval, _ = jax.vmap(evaluate_all, in_axes=(0, None))(x_batch, weights)
        # TODO: It's not clear that mean is the best here.
        return jnp.mean(batch_eval), batch_eval

    def verify_batch(x_batch: Array) -> Array:
        batch_res = jax.vmap(verify_all)(x_batch)
        return batch_res

    return evaluate_batch, verify_batch


# TODO: Enable solver switching based on command line.
def solver_config(solver: str, eval_fun: Callable):
    match solver:
        case "lbfgsb":
            return LBFGSB(fun=eval_fun, maxiter=500)#, linesearch_init="current")
        case "pgd":
            return ProjectedGradient(fun=eval_fun, projection=projection_box, maxiter=50, has_aux=True)
        case "prox":
            return ProximalGradient()


# TODO: Enable starting point selection strategies on command line (will subsume vertex start and fuzzing)
def next_x0_strategy():
    pass

# (241, 3) (241, 4, 1) (241, 4)

# TODO: Add "solver" and "selection" params? For now L-BFGS-B is the best anyway.
def run_solver(
    timeout: int,
    n_vars: int,
    batch_sz: int,
    restart: int,
    fuzz_limit: int,
    opt_combined: bool,
    objs: tuple[Objective, ...],
    v_start: bool = False,
    n_devices: int = 1,
) -> float:
    # Create device mesh, declare JAX specs, and shard:
    # tsolver = time()
    # print("Solver Init")
    # TODO: enabled parameterisation for TPU?
    devices = jax.devices("gpu")[:n_devices]
    device_batch = batch_sz * len(devices)
    weights = [jnp.ones_like(obj.clauses.types, dtype=jnp.bfloat16) for obj in objs]
    mesh = Mesh(devices, ("device",))
    shard_spec = P("device")
    restart = np.inf if not restart else restart

    best_unsat = jnp.inf
    best_x = None
    # best_start = None

    all_iters = []
    all_unsat_cts = []
    starts = 0
    restart_ct = 0
    key = jax.random.PRNGKey(restart_ct)
    fuzz_key = jax.random.PRNGKey(restart_ct + 1)
    timeout_m, timeout_s = divmod(timeout, 60)
    # print("Complete Init", time() - tsolver)
    t0 = time()

    with mesh:
        # print("Before shardings", time() - tsolver)
        clause_sharding = NamedSharding(mesh, shard_spec)
        sharded_objs = tuple(shard_objective(obj, clause_sharding) for obj in objs)
        sharded_weight = [jax.device_put(weight, clause_sharding) for weight in weights]
        # print("After shardings", time() - tsolver)

        if opt_combined:
            evaluator, verifier = eval_verify_combined(sharded_objs)
        else:
            evaluator, verifier = eval_verify(sharded_objs)
        solver = solver_config("lbfgsb", evaluator)  # lbfgsb, pgd

        # VERIFY WEIGHTS?!?!?
        def optimize(x: Array, weights: list[Array]) -> tuple[Array, Array, Array]:
            print(f"TRACING optimize with shapes: {opt_combined} {x.shape}")
            x_opt, state = solver.run(x, bounds=(-1 * jnp.ones_like(x), jnp.ones_like(x)), weights=weights)  # L-BFGS-B
            unsat = jnp.squeeze(verifier(x_opt))
            return x_opt, unsat, jnp.atleast_1d(state.iter_num), state.aux

        def process_batch_combined(x_batch: Array, weights: list[Array]) -> tuple[Array, Array, Array, Array]:
            optimize_batch = optimize
            x_opt, unsat, iters, eval_scores = optimize_batch(x_batch, weights)
            unsat_cl_count = jnp.sum(jnp.atleast_1d(unsat), axis=0)
            return x_opt, unsat, iters, unsat_cl_count, eval_scores

        def process_batch(x_batch: Array, weights: list[Array]) -> tuple[Array, Array, Array, Array]:
            print("VMAP optimize")
            optimize_batch = jax.vmap(optimize, in_axes=(0,))
            x_shard = jax.lax.with_sharding_constraint(x_batch, clause_sharding)
            x_opt, unsat, iters, eval_scores = optimize_batch(x_shard, weights)
            unsat_cl_count = jnp.sum(jnp.atleast_1d(unsat), axis=0)
            return x_opt, unsat, iters, unsat_cl_count, eval_scores

        # print("JIT the whole thing", time() - tsolver)
        process = process_batch_combined if opt_combined else process_batch
        opt_batch = jax.jit(process, in_shardings=(None, [clause_sharding] * len(weights)))
        # print("JITTED", time() - tsolver)

        # Precompile
        print("PRECOMPILE")
        opt_batch(jnp.zeros((device_batch, n_vars)), sharded_weight)
        print("PRECOMPILE FINISHED")

        batch_time = t0
        pbar = tqdm(
            total=timeout,
            desc=f"restart {restart_ct} ({starts}/{restart} -- best={best_unsat})",
            bar_format="{l_bar}{bar}| {elapsed}/" + str(timeout_m).zfill(2) + ":" + str(timeout_s).zfill(2),
        )
        while time() - t0 < timeout:
            # fh = logging.FileHandler(filename="jax_{}_{}.log".format(restart_ct, starts))
            # Remove any existing handlers
            # for handler in jaxlog.handlers[:]:
            #     jaxlog.removeHandler(handler)
            # jaxlog.addHandler(fh)

            tloop = time()
            # print("##### New Loop start || elapsed since call to run:", tloop - tsolver)
            key, subkey = jax.random.split(key)
            fuzz_key, subfuzz_key = jax.random.split(fuzz_key)

            sample_method = "uniform"  # or 'bias' or 'coin' (weighted benoulli coin)

            match sample_method:
                case "bias":
                    # Generate values biased towards 1 (False) while maintaining full [-1, 1] range
                    # Method 1: Using power transformation with bias_strength (original)
                    u = jax.random.uniform(subkey, minval=0.0, maxval=1.0, shape=(device_batch, n_vars))
                    bias_strength = 0.5  # Lower value = stronger bias towards 1 (values 0-1 the domain servi)
                    x0 = 2 * u**bias_strength - 1
                case "coin":
                    # Method 2: Using a biased coin flip (Bernoulli distribution)
                    # Uncomment to use this method instead
                    false_prob = 0.7  # Y% probability of generating values tending towards False
                    coin_key, subkey = jax.random.split(subkey)
                    biased_coins = jax.random.bernoulli(coin_key, p=false_prob, shape=(device_batch, n_vars))
                    signs = 2 * biased_coins - 1  # Convert to -1/1
                    magnitudes = jax.random.uniform(subkey, minval=0.0, maxval=1.0, shape=(device_batch, n_vars))
                    x0 = signs * magnitudes
                case "uniform" | _:
                    x0 = jax.random.uniform(subkey, minval=-1, maxval=1, shape=(device_batch, n_vars))

            fuzz = jax.random.uniform(subfuzz_key, minval=1e-10, maxval=1e-4, shape=x0.shape)
            fuzz_attempt = 0

            if v_start:
                # snap to nearest vertex and fuzz to avoid immediate saddle
                x0 = jnp.sign(x0)
                x0 = x0 + -x0 * fuzz
            # x_starts = x0[:]
            # print("Random keys regen & snap", time() - tloop)

            # Do it.
            print("Before OPT call", time() - tloop)
            # with jax.profiler.trace(f"/tmp/jax-trace-restart-{restart_ct}", create_perfetto_link=True):
            #     opt_x0, opt_unsat, opt_iters, opt_unsat_ct, eval_scores = opt_batch(x0, sharded_weight)
            opt_x0, opt_unsat, opt_iters, opt_unsat_ct, eval_scores = opt_batch(x0, sharded_weight)
            print("After OPT call", time() - tloop)
            batch_unsat_scores = jnp.sum(jnp.atleast_2d(opt_unsat), axis=1)
            print("unsat scores:", batch_unsat_scores, "\n eval scores:", eval_scores)
            batch_best = jnp.min(batch_unsat_scores)
            print("eval of best score:", batch_best, jnp.max(eval_scores))
            best_unsat = batch_best if batch_best < best_unsat else best_unsat
            # print("After some JNP + maybe fuzz next", time() - tloop)

            found_sol = False
            if batch_best == 0:
                print("Found a solution!")
                found_sol = True

            elif fuzz_limit:
                xF = opt_x0[:]
                # Solution not found - fuzz current convergence.
                print("BEFORE:", opt_x0[0][:5])
                fuzz_key, subfuzz_key = jax.random.split(fuzz_key)
                fuzz = jax.random.uniform(subfuzz_key, minval=1e-7, maxval=1e-2, shape=x0.shape)
                fuzz_mag = 1
                while fuzz_attempt < fuzz_limit:
                    fuzz_attempt += 1
                    xFC = xF[:]
                    flip = xF + fuzz ** (1 / fuzz_mag)
                    flip = jnp.abs(flip) > 1
                    # fuzz = jnp.where(flip, -fuzz, fuzz)
                    # xF = xF + -jnp.sign(xF) * fuzz**fuzz_mag
                    fuzz = jnp.sign(fuzz) * jnp.abs(fuzz) ** (1 / fuzz_mag)
                    print(fuzz[0][0:5])
                    xF = jnp.clip(xF + fuzz, -1, 1)
                    print(f"FUZZ {fuzz_mag}:", xF[0][:5])
                    print("Before OPT+FUZZ call", time() - tloop)
                    opt_xF, opt_unsatF, opt_itersF, opt_unsat_ctF, eval_scoresF = opt_batch(xF, sharded_weight)
                    print("After OPT+FUZZ call", time() - tloop)
                    batch_unsatF_scores = jnp.sum(jnp.atleast_2d(opt_unsatF), axis=1)
                    batch_bestF = jnp.min(batch_unsatF_scores)
                    # print("JNP stuff on FUZZ results", time() - tloop)
                    if batch_bestF < best_unsat:
                        best_unsat = batch_bestF
                        opt_x0, opt_unsat, opt_iters, opt_unsat_ct = xF, opt_unsatF, opt_itersF, opt_unsat_ctF
                    if batch_bestF == 0:
                        print(f"Fuzz {fuzz_attempt} found a solution!")
                        found_sol = True
                        break
                    if (jnp.sign(xFC) == jnp.sign(opt_xF)).all():
                        print("INCREASE MAGNITUDE")
                        fuzz_mag += 1
                    else:
                        mymask = jnp.nonzero(jnp.sign(xFC) != jnp.sign(opt_xF))
                        fnz = (mymask[0][0], mymask[1][0])
                        print(
                            f"diff at pos {fnz}, start {xFC[fnz]} + fuzz {fuzz[fnz]} = {xF[fnz]}. Descended to {opt_xF[fnz]}"
                        )
                        fuzz_mag = 1
                    xF = opt_xF
                    # print(f"Optim Iters: min: {jnp.min(opt_itersF)}, max: {jnp.max(opt_itersF)}, mean: {jnp.mean(opt_itersF)}")
            # print(fuzz_limit, "Fuzz complete", time() - tloop)

            loc = jnp.argmin(batch_unsat_scores)
            best_x = opt_x0[loc]
            best_unsat_clause_idx = jnp.nonzero(jnp.atleast_2d(opt_unsat)[loc])
            # print("Stats gather", time() - tloop)
            # clauses = {}
            # for
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
                print(f"Optim Iters: min: {jnp.min(opt_iters)}, max: {jnp.max(opt_iters)}, mean: {jnp.mean(opt_iters)}")
                print(opt_iters, eval_scores, opt_unsat, opt_x0[-3])
                break
            print(f"Optim Iters: min: {jnp.min(opt_iters)}, max: {jnp.max(opt_iters)}, mean: {jnp.mean(opt_iters)}")

            all_iters.append(opt_iters)
            all_unsat_cts.append(opt_unsat_ct)
            starts += 1
            if starts >= restart:
                # break
                # print("Performing restart", time() - tloop)
                # print("Memory before restart:", jax.devices()[0].memory_stats())
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

                penalty = unsat_ct.sum(axis=0)
                worst = penalty.max()

                print(
                    f"Unsat counts: {penalty}, \nObjs: {[obj.clauses.lits.shape for obj in objs]}, \nBest: {best_unsat_clause_idx}"
                )
                pen_start = 0
                # adjust to always target -k?
                for idx, weight in enumerate(weights):
                    n_clause = len(weight)
                    pen_end = pen_start + n_clause
                    w_pens = penalty[pen_start:pen_end]
                    adj_weight = jnp.where(w_pens > 0, weight + 0.1 * w_pens / worst, 1)
                    sharded_weight[idx] = jax.device_put(adj_weight, clause_sharding)
                    pen_start += n_clause
                # tx = time()
                # print(time() - tx, "SEP time to reload weights")

                stamp = time()
                print("Restart time: {:.2f}/{}".format(stamp - t0, timeout))
                # print("Memory after restart:", jax.devices()[0].memory_stats())
                # print("Restart complete", time() - tloop)

            pbar.set_description(f"restart {restart_ct} ({starts}/{restart} -- best={best_unsat})")
            pbar.update(time() - batch_time)
            batch_time = time()
            # print("end loop", time() - tloop)
        pbar.close()
    return time() - t0


def main(
    file: str, mode: int, timeout: int, batch: int, restart: int, fuzz: int, combine: bool, vertex: bool, n_devices: int
) -> None:
    if file is None:
        print("Error: Please provide a (hybrid) dimacs CNF file")
        return 1

    sat = Formula()
    stamp1 = time()
    sat.read_DIMACS(file)
    stamp2 = time()
    read_time = stamp2 - stamp1

    n_vars = sat.n_var
    objectives, _ = process_clauses(sat, mode, n_devices=n_devices)
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
    n_devices = len(jax.devices("gpu"))
    ap = ArgParse(description="Process a file with optional parameters", 
                  epilog="Some debug options:" + 
                  "JAX_COMPILER_DETAILED_LOGGING_MIN_OPS=0" +
                  "JAX_LOGGING_LEVEL=DEBUG TF_CPP_MIN_LOG_LEVEL=0" + 
                  "JAX_TRACEBACK_FILTERING=off")
    ap.add_argument("file", help="The file to process")
    ap.add_argument("-p", "--profile", action="store_true", help="Enable profiling")
    ap.add_argument("-t", "--timeout", type=int, default=300, help="Maximum runtime (timeout seconds)")
    ap.add_argument("-b", "--batch", type=int, default=16, help="Batch size")
    ap.add_argument("-f", "--fuzz", type=int, default=0, help="Number of times to attempt fuzzing per batch")
    ap.add_argument("-v", "--vertex", action="store_true", help="Start near vertices")
    ap.add_argument("-c", "--combine", action="store_true", help="Jointly optimise entire batches")
    ap.add_argument("-r", "--restart", type=int, default=0, help="Batches before reweight and restart (never if 0)")
    ap.add_argument("-m", "--mode", type=int, default=0, help="Clause partitioning mode. Provide 0 to be asked")
    ap.add_argument("-n", "--n_devices", type=int, default=n_devices, help="Devices (eg. GPUs) to use. 0 uses all")

    arg = ap.parse_args()
    if not arg.n_devices:
        arg.n_devices = len(jax.devices())

    # Run with or without profiler based on the flag
    profiler = jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True) if arg.profile else nullcontext()
    with profiler:
        main(arg.file, arg.mode, arg.timeout, arg.batch, arg.restart, arg.fuzz, arg.combine, arg.vertex, arg.n_devices)
