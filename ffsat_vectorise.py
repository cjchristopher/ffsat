from __future__ import annotations

import functools
import logging
import os
import sys
from argparse import ArgumentParser as ArgParse
from contextlib import nullcontext
from time import perf_counter as time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable pre-allocation
os.environ["XLA_FLAGS"] = " ".join(
    [
        "--xla_enable_fast_math=true",
        "--xla_gpu_triton_gemm_any=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_highest_priority_async_stream=true",
        "--xla_gpu_enable_fast_min_max=true",
        "--xla_gpu_enable_cublaslt=true",
        "--xla_gpu_autotune_gemm_rtol=1e-6",
        "--xla_gpu_exhaustive_tiling_search=true",
        # "--xla_gpu_deterministic_ops=true",
        "--xla_gpu_require_complete_aot_autotune_results=true",
    ]
)

# Single-host, multi-device computation on NVIDIA GPUs
# os.environ.update({
#   "NCCL_LL128_BUFFSIZE": "-2",
#   "NCCL_LL_BUFFSIZE": "-2",
#    "NCCL_PROTO": "SIMPLE,LL,LL128",
#  })

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding
from typing import Optional as Opt
from typing import TypeAlias
from tqdm import tqdm

from boolean_whf import Objective
from sat_loader import PBSATFormula
from solvers import FFSatSolver, build_eval_verify, seq_eval_verify

jax.config.update("jax_platform_name", "gpu")  # gpu/cpu/tpu
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_use_shardy_partitioner", True)
jax.config.update("jax_memory_fitting_level", "O3")
jax.config.update("jax_optimization_level", "O3")
jax.config.update("jax_compiler_enable_remat_pass", False)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
#jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

# # DEBUGGING BLOCK
# jax.config.update("jax_debug_nans", True)
jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_no_tracing", True)
# jax.config.update("jax_disable_jit", True)
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

print = functools.partial(print, flush=True)
ShardSpec: TypeAlias = Opt[tuple[NamedSharding, tuple[NamedSharding, ...]]]

def x0_guesses(subkey: Array, batch_sz: int, n_vars: int, method: str = "bias") -> Array:
    """
    Generates initial guesses for variable assignments in SAT problems using different randomization methods.

    Args:
        subkey (Array): JAX PRNG key for random number generation.
        batch_sz (int): Number of guess vectors to generate.
        n_vars (int): Number of variables in each guess vector.
        method (str, optional): Method for generating guesses. Options are:
            - "bias" (default): Generates values biased towards False from a uniform distribution
            - "coin": Generates using a biased (70% tending False) coin flip (Bernoulli).
            - "uniform": Generates values uniformly between True and False

    Returns:
        Array: An array of shape (batch_sz, n_vars) containing the generated initial guesses.

    Raises:
        ValueError: If an unsupported method is specified.
    """

    if method == "bias":
        # Generate values biased towards 1 (False) while maintaining full [-1, 1] range
        u = jax.random.uniform(subkey, minval=0.0, maxval=1.0, shape=(batch_sz, n_vars))
        bias_strength = 0.5  # Lower value = stronger bias towards False
        return 2 * u**bias_strength - 1

    elif method == "coin":
        false_prob = 0.7  # Y% probability of generating values tending towards False
        coin_key, subkey = jax.random.split(subkey)
        biased_coins = jax.random.bernoulli(coin_key, p=false_prob, shape=(batch_sz, n_vars))
        signs = 2 * biased_coins - 1
        magnitudes = jax.random.uniform(subkey, minval=0.0, maxval=1.0, shape=(batch_sz, n_vars))
        return signs * magnitudes

    elif method == "uniform":
        return jax.random.uniform(subkey, minval=-1, maxval=1, shape=(batch_sz, n_vars))

    else:
        pass

def shard_objective(target: Objective | tuple[Array], sharding: NamedSharding) -> Array | tuple[Array]:
    mesh = sharding.mesh
    replication = NamedSharding(mesh, jax.P())

    def shard_leaf(leaf):
        if isinstance(leaf, jax.Array):
            # Replicate if scalar or if the first dimension has size 1
            if leaf.ndim == 0 or (leaf.ndim > 0 and leaf.shape[0] == 1):
                return jax.device_put(leaf, replication)
            # Otherwise, shard along the first dimension
            return jax.device_put(leaf, sharding)
        return leaf

    return jax.tree_util.tree_map(shard_leaf, target)


def mesh_batch(devices: list, n_vars: int, objs: tuple[Objective], batch: int) -> tuple[Mesh, ShardSpec, int]:
    """ 
    By default we prefer more points, so always a (1, n_gpu) mesh. For truly enormous objectives we can always
    do (n_gpu, 1) and only treat a single batch of points, otherwise if the objectives are still quite big, for 
    nice grid sizes other splits can be considered (heuristically?) e.g:
    4 GPU - 50/50 at (2,2), 6 GPU - (3,2) or (2,3), 8 GPU - (4,2) or (2,4), etc...
    """
    # TODO: We could probably stick the batch size calculation in here as well, since it will depend on this decision
    # and just pass use_sharding in as a parameter and skip the first section. We would need to pass the batch_size
    # anyway if user specified for the heuristics?

    # If no batch size has been provided, estimate a reasonable maximum throughput based on the largest obj size.
    dt_sz = jnp.dtype(objs[0].ffts.dft.dtype).itemsize
    obj_mem = max([np.prod([max(obj.clauses.lits.shape), max(obj.ffts.dft.shape) ** 2, dt_sz]) for obj in objs])
    max_gpu_mem = devices[0].memory_stats()["bytes_limit"] - devices[0].memory_stats()["pool_bytes"]
    print(
        f"GPU: {round(max_gpu_mem / (1024**3), 2)} ({round(max_gpu_mem / 1e9, 2)}) GiB",
        f"\nObj mem: {round(obj_mem / 1024, 2)} KiB",
    )
    bitshift = (int(np.floor(max_gpu_mem / (obj_mem))) * n_devices).bit_length()
    opt_batch = (1 << bitshift) - (1 << max(0, bitshift - 3))
    if batch == -1 or batch > opt_batch:
        print("Setting batch size to", opt_batch, "from", batch)
        batch = opt_batch

    #     n_dev = len(devices)
    #     clause_dim = 1
    #     batch_dim = 1
    #     if n_dev > 1:
    #         gpu_mem = devices[0].memory_stats()['bytes_limit']
    #         obj_gpu_bytes = 0
    #         for obj in objectives:
    #             obj_gpu_bytes += sum([leaf.nbytes for leaf in jax.tree.leaves(obj) if isinstance(leaf, Array)])
    #         # For large enough ratio, compute the number of gpus required to bring objectives back below 25%
    #         # the mesh the rest to assignment batch.
    #         if (obj_gpu_bytes / gpu_mem > 0.25) and (n_dev >= 4):
    #             # Objectives are worth distributing
    #             clause_dim = n_dev//2
    #             batch_dim = (n_dev - n_dev%2)/clause_dim

    mesh = Mesh(np.array(devices).reshape((1, len(devices))), ("objective", "batch"))
    with jax.sharding.use_mesh(mesh):
        objective_sharding = NamedSharding(mesh, jax.P("objective"))
        batch_sharding = NamedSharding(mesh, jax.P("batch"))
        shard_spec = (batch_sharding, tuple(objective_sharding for _ in objs))

    return mesh, shard_spec, batch


def run_solver(
    timeout: int,
    n_vars: int,
    n_clause: int,
    batch: int,
    restart: int,
    fuzz_limit: int,
    objectives: tuple[Objective, ...],
    n_devices: int = 1,
    sample_method: str = "bias",
    sol_name: str = "lbfgsb",
    use_sharding: bool = True,
    warmup: bool = True,
) -> float:
    devices = jax.devices("gpu")[:n_devices]
    best_unsat = jnp.inf
    best_x = None

    all_dcnt_iters = []
    all_unsat_cts = []
    starts = 0
    restart_ct = 0
    timeout_m, timeout_s = divmod(timeout, 60)
    mesh, shard_spec, batch = mesh_batch(devices, n_vars, objectives, batch)

    t0 = time()
    with jax.sharding.use_mesh(mesh):
        key = jax.random.PRNGKey(np.array(restart_ct))
        fuzz_key = jax.random.PRNGKey(np.array(restart_ct + 1))

        batch_sharding, (objective_sharding, *_) = shard_spec
        objectives = tuple(shard_objective(obj, objective_sharding) for obj in objectives)
        weights = tuple(jnp.full((obj.clauses.lits.shape[0],), 1.0, dtype=float) for obj in objectives)
        weights = tuple(shard_objective(weight, objective_sharding) for weight in weights)

        obj_evaluators, obj_verifiers = build_eval_verify(objectives)
        evaluator, verifier = seq_eval_verify(obj_evaluators, obj_verifiers)

        warmup_data = None
        if warmup:
            w_pre = tuple(jnp.full((obj.clauses.lits.shape[0],), 0.9999, dtype=float) for obj in objectives)
            x_pre = jnp.full((batch, n_vars), fill_value=-1, dtype=float)
            x_shard_pre = jax.device_put(x_pre, batch_sharding)
            w_shard_pre = tuple(shard_objective(w_pre_obj, objective_sharding) for w_pre_obj in w_pre)
            jax.debug.visualize_array_sharding(x_shard_pre)
            warmup_data = (x_shard_pre, w_shard_pre)

        solver = FFSatSolver(evaluator, verifier, sol_name, shard_spec, warmup_data)

        t0 = time()
        batch_time = t0
        pbar = tqdm(
            total=timeout,
            desc=f"restart {restart_ct} ({starts}/{restart} -- best={best_unsat})",
            bar_format="{l_bar}{bar}| {elapsed}/" + str(timeout_m).zfill(2) + ":" + str(timeout_s).zfill(2),
        )

        while time() - t0 < timeout:
            tloop = time()
            key, subkey = jax.random.split(key)
            fuzz_key, subfuzz_key = jax.random.split(fuzz_key)

            x0 = x0_guesses(subkey, batch, n_vars, sample_method)
            fuzz = jax.random.uniform(subfuzz_key, minval=1e-10, maxval=1e-4, shape=x0.shape)
            fuzz_attempt = 0

            # Do it.
            print("\nBefore OPT call", time() - tloop)
            if use_sharding:
                x0_shard = jax.device_put(x0, batch_sharding)
            else:
                x0_shard = jax.device_put(x0, devices[0])

            opt_x0, opt_unsat, opt_iters, opt_unsat_ct, eval_scores = solver.run(x0_shard, weights)
            print("After OPT call", time() - tloop)

            batch_unsat_scores = jnp.sum(jnp.atleast_2d(opt_unsat), axis=1)
            batch_best = jnp.min(batch_unsat_scores)
            best_unsat = batch_best if batch_best < best_unsat else best_unsat
            #print("unsat scores:", batch_unsat_scores.shape, batch_unsat_scores)  # , "\n eval scores:", eval_scores)
            #print("eval of best score:", batch_best, jnp.min(eval_scores))
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
                    opt_xF, opt_unsatF, opt_itersF, opt_unsat_ctF, eval_scoresF = solver.run(xF, weights)
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
                print(x0_shard[loc])
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
                print("More stats", opt_iters, opt_iters[loc], eval_scores, opt_unsat, opt_x0[-3])
                break
            print(f"Optim Iters: min: {jnp.min(opt_iters)}, max: {jnp.max(opt_iters)}, mean: {jnp.mean(opt_iters)}")

            all_dcnt_iters.append(opt_iters)
            all_unsat_cts.append(opt_unsat_ct)
            starts += 1
            if restart and starts >= restart:
                iters = jnp.concat(all_dcnt_iters)
                unsat_ct = jnp.array(all_unsat_cts)

                # Reset/update for next restart
                restart_ct += 1
                key = jax.random.PRNGKey(np.array(restart_ct))
                fuzz_key = jax.random.PRNGKey(np.array(restart_ct + 1))
                starts = 0
                all_dcnt_iters = []
                all_unsat_cts = []
                penalty = unsat_ct.sum(axis=0)
                worst = penalty.max()

                print(f"Optim Iters: min: {jnp.min(iters)}, max: {jnp.max(iters)}, mean: {jnp.mean(iters)}")
                print(f"Restarts: {restart_ct} | Current  (#unsat): {best_unsat}")
                print(f"Unsat counts: {penalty}, \nBest: {best_unsat_clause_idx}")

                pen_start = 0
                # adjust to always target -k?
                for idx, weight in enumerate(weights):
                    n_clause = len(weight)
                    pen_end = pen_start + n_clause
                    w_pens = penalty[pen_start:pen_end]
                    adj_weight = jnp.where(w_pens > 0, weight + 0.1 * w_pens / worst, 1)
                    weights[idx] = jax.device_put(adj_weight, objective_sharding)
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
    file: str,
    timeout: int,
    batch: int | None,
    restart: int,
    fuzz: int,
    n_devices: int,
    disk_cache: str = None,
) -> None:
    if file is None:
        print("Error: Please provide a (hybrid) dimacs CNF file")
        return 1

    sat_parser = PBSATFormula(max_workers=4, n_devices=n_devices, disk_cache=disk_cache)
    stamp1 = time()
    sat_parser.read_DIMACS(file)
    stamp2 = time()
    read_time = stamp2 - stamp1

    n_var = sat_parser.n_var
    n_clause = sat_parser.n_clause
    objectives = sat_parser.process_clauses()
    stamp1 = time()
    process_time = stamp1 - stamp2

    t_solve = run_solver(timeout, n_var, n_clause, batch, restart, fuzz, objectives, n_devices=n_devices)
    print("Some stats")
    print("Time reading input:", read_time)
    print("Time processing to Arrays:", process_time)
    print("Time spent solving:", t_solve)


if __name__ == "__main__":
    n_devices = len(jax.devices("gpu"))
    ap = ArgParse(
        description="Process a file with optional parameters",
        epilog="Some debug options:"
        + "JAX_COMPILER_DETAILED_LOGGING_MIN_OPS=[X]"
        + "JAX_LOGGING_LEVEL=DEBUG TF_CPP_MIN_LOG_LEVEL=[X] TF_CPP_MAX_VLOG_LEVEL=[X]"
        + "JAX_TRACEBACK_FILTERING=off",
    )
    LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    ap.add_argument("file", help="The file to process")
    ap.add_argument("-p", "--profile", action="store_true", help="Enable profiling")
    ap.add_argument("-t", "--timeout", type=int, default=300, help="Maximum runtime (timeout seconds)")
    ap.add_argument("-b", "--batch", type=int, default=-1, help="Batch size. -1 computes maximum for hardware")
    ap.add_argument("-f", "--fuzz", type=int, default=0, help="Number of times to attempt fuzzing per batch")
    ap.add_argument("-r", "--restart", type=int, default=0, help="Batches before reweight and restart (never if 0)")
    ap.add_argument("-n", "--n_devices", type=int, default=n_devices, help="Devices (eg. GPUs) to use. 0 uses all")
    ap.add_argument("-d", "--debug", choices=LOG_LEVELS, default="ERROR", help=f"Set logging level ({LOG_LEVELS})")

    arg = ap.parse_args()
    if not arg.n_devices:
        arg.n_devices = len(jax.devices())

    logging.basicConfig(
        level=getattr(logging, arg.debug.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            # Optional: logging.FileHandler('sat_loader.log')  # Also log to file
        ],
    )
    # Run with or without profiler based on the flag
    profiler = jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True) if arg.profile else nullcontext()
    with profiler:
        main(arg.file, arg.timeout, arg.batch, arg.restart, arg.fuzz, arg.n_devices)
