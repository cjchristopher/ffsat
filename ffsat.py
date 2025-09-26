from __future__ import annotations

import functools
import logging
import math
import os
import sys
from argparse import ArgumentParser as ArgParse
from contextlib import nullcontext
from time import perf_counter as time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable pre-allocation
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.95"  # Use full memory allocation
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
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
os.environ.update({
  "NCCL_LL128_BUFFSIZE": "-2",
  "NCCL_LL_BUFFSIZE": "-2",
   "NCCL_PROTO": "SIMPLE,LL,LL128",
 })

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, AxisType
from typing import Optional as Opt
from typing import TypeAlias
from tqdm import tqdm

from contextlib import nullcontext
from boolean_whf import Objective
from sat_loader import PBSATFormula
from solvers import FFSatSolver, build_eval_verify, seq_eval_verify

# TODO: Disable x64 when clauses are short enough - find this limit.
jax.config.update("jax_platform_name", "gpu")  # gpu/cpu/tpu
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
jax.config.update("jax_use_shardy_partitioner", True)
jax.config.update("jax_memory_fitting_level", "O3")
jax.config.update("jax_optimization_level", "O3")
jax.config.update("jax_compiler_enable_remat_pass", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
#jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

# # DEBUGGING BLOCK
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_log_compiles", True)
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

def x0_guesses(rng_key: Array, batch: int, n_vars: int, method: str = "bias", prefixes: Array = None) -> Array:
    """
    Generates initial guesses for variable assignments in SAT problems using different randomization methods.

    Args:
        rng_key (Array): JAX PRNG key for random number generation.
        batch (int): Number of guess vectors to generate.
        n_vars (int): Number of variables in each guess vector.
        method (str, optional): Method for generating guesses. Options are:
            - "bias" (default): Generates values biased towards False from a uniform distribution
            - "coin": Generates using a biased (70% tending False) coin flip (Bernoulli).
            - "uniform": Generates values uniformly between True and False
        prefix_vectors (Array, optional): Shape (N, n_vars) where 0=no fix, ±1=fix to that value.
                                        Will be replicated to fill batch size B, so each vector appears B//N times.

    Returns:
        Array: An array of shape (batch, n_vars) containing the generated initial guesses.

    Raises:
        ValueError: If an unsupported method is specified.
    """

    if method == "bias":
        # Generate values biased towards 1 (False) while maintaining full [-1, 1] range
        u = jax.random.uniform(rng_key, minval=0.0, maxval=1.0, shape=(batch, n_vars))
        bias_strength = 0.5  # Lower value = stronger bias towards False
        x0 = 2 * u**bias_strength - 1

    elif method == "coin":
        false_prob = 0.7  # Y% probability of generating values tending towards False
        coin_key, rng_key = jax.random.split(rng_key)
        biased_coins = jax.random.bernoulli(coin_key, p=false_prob, shape=(batch, n_vars))
        signs = 2 * biased_coins - 1
        magnitudes = jax.random.uniform(rng_key, minval=0.0, maxval=1.0, shape=(batch, n_vars))
        x0 = signs * magnitudes

    elif method == "uniform":
        x0 = jax.random.uniform(rng_key, minval=-1, maxval=1, shape=(batch, n_vars))

    else:
        raise ValueError(f"Unsupported method: {method}")

    # Fix positions of supplied prefixes from classical SAT solver.
    if prefixes is not None:
        N = prefixes.shape[0]
        # Batch is already correctly sized equal points for each prefix.
        replicated_prefixes = jnp.repeat(prefixes, batch // N, axis=0)

        # Non-zero points are fixed, so adjust batch and disable gradients there.
        fixed_mask = (replicated_prefixes != 0)
        x0 = jnp.where(fixed_mask, replicated_prefixes, x0)
        x0 = jnp.where(fixed_mask, jax.lax.stop_gradient(x0), x0)

    return x0


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


def get_mesh(devices: list) -> tuple[Mesh, ShardSpec]:
    """ 
    By default we prefer more points, so always a (1, n_gpu) mesh. For truly enormous objectives we can always
    do (n_gpu, 1) and only treat a single batch of points, otherwise if the objectives are still quite big, for 
    nice grid sizes other splits can be considered (heuristically?) e.g:
    4 GPU - 50/50 at (2,2), 6 GPU - (3,2) or (2,3), 8 GPU - (4,2) or (2,4), etc...

    NB SEPT-25: Actually, I think we should just always maximize points per GPU and never shard the objectives.
    We have the capability to here, of course, but unless the problem is truly enormous, I think it's probably likely
    that higher throughput will still be achieved only sharding in the batch.
    """
    # The reshape here would be adjusted if we ever wanted to shard over objectives as well.
    mesh = Mesh(np.array(devices).reshape((1, len(devices))), ("objective", "batch"))
    jax.sharding.set_mesh(mesh)

    objective_spec = jax.P("objective")
    objective_sharding = NamedSharding(mesh, objective_spec)

    batch_spec = jax.P("batch")
    batch_sharding = NamedSharding(mesh, batch_spec)

    return mesh, objective_sharding, batch_sharding


def adjust_batch(devices: list, batch: int, est_mem_per_point: int, n_prefix: int = 1) -> int:
    n_device = len(devices)
    max_gpu_mem = devices[0].memory_stats()["bytes_limit"] - devices[0].memory_stats()["bytes_in_use"]*2
    max_batch = max_gpu_mem//est_mem_per_point

    pad = " "*(len(str(batch))+4)
    #print(pad + "Mem per batch element:", max_gpu_mem, max_batch, "(", devices[0].memory_stats()["bytes_in_use"]*2)

    if batch == -1 or batch > max_batch:
        print("Adjusting per-device batch size (either none specified to batch too large):")
        print(f"Set to {max_batch} p.d. (total {max_batch * n_device}) from {batch} (theoretical max {max_batch})")
        batch = (max_batch//2) * n_device
    else:
        batch = batch * n_device

    # Adjust alignment.
    alignment = math.lcm(n_prefix, n_device)
    opt_batch = (batch // alignment) * alignment

    # Ensure we have at least one batch element per prefix and device combination
    if opt_batch < alignment:
        opt_batch = alignment
        print(f"Warning: Batch size set to minimum {opt_batch} to fit {n_prefix} prefixes and {n_device} devices")

    if n_prefix > 1:
        points_per_prefix = opt_batch // n_prefix
        print(f"Batch distribution: {opt_batch} total points = {n_prefix} prefixes x {points_per_prefix} points each")

    return opt_batch


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
    sol_name: str = "pgd",
    use_sharding: bool = True,
    warmup: bool = False,
    benchmark: bool = False,
    counting: int = 0,
    rand_seed: bool = False,
    prefix_vectors: Array = None,
    q_bench: int = 0,
) -> float:
    devices = jax.devices("gpu")[:n_devices]

    mesh, objective_sharding, batch_sharding = get_mesh(devices)
    jax.sharding.set_mesh(mesh)

    # Construct weights, and shard both weights and objectives.
    objectives = tuple(shard_objective(obj, objective_sharding) for obj in objectives)
    weights = tuple(jnp.full((obj.clauses.lits.shape[0],), 1.0, dtype=float) for obj in objectives)
    weights = tuple(shard_objective(weight, objective_sharding) for weight in weights)

    # Construct pure JAX functions (closures) and build solver.
    obj_evaluators, obj_verifiers = build_eval_verify(objectives)
    evaluator, verifier = seq_eval_verify(obj_evaluators, obj_verifiers)
    solver = FFSatSolver(evaluator, verifier, sol_name, maxiter=20, bench=benchmark)

    seed = int(time()) if rand_seed else 0
    key = jax.random.PRNGKey(np.array(seed))
    f_key = jax.random.PRNGKey(np.array(seed + 1))

    if batch == -1:
        # User has requested we auto-select optimal batch size.
        # Use objective sizes to get initial guess for maximum batch size.
        print("Guessing optimal batch size (e.g. none specified):")
        total_gpu_mem = devices[0].memory_stats()["bytes_limit"]
        dt_sz = jnp.dtype(objectives[0].ffts.dft.dtype).itemsize
        total_obj_mem = sum([np.prod([max(o.clauses.lits.shape), max(o.ffts.dft.shape) ** 2, dt_sz]) for o in objectives])
        batch = int(np.floor(total_gpu_mem / (total_obj_mem)))
        batch += batch % 2

    # Generate arrays and check peak memory estimation.
    if q_bench < 2: # 2 implies we have already worked out the upper limit
        noise = jax.random.uniform(f_key, maxval=5e-2, shape=(batch, n_vars))
        dummy_batch = batch
        x_dummy = jax.device_put(jnp.full((batch, n_vars), fill_value=0.99, dtype=float) - noise, batch_sharding)
        w_dummy = [w - 0.00001 for w in weights]
        peak_mem = solver.peak_memory_estimation((x_dummy, w_dummy))
        mem_est_per_point = peak_mem // batch
    else:
        mem_est_per_point = 1

    # Adjust batch for max throughput, fairness, and to ensure we can allocate.
    n_prefix = len(prefix_vectors) if prefix_vectors is not None else 1
    batch = adjust_batch(devices, batch, mem_est_per_point, n_prefix)

    if q_bench == 1:
        return batch
        #pass

    if warmup:
        if q_bench == 2 or dummy_batch != batch:
            noise = jax.random.uniform(f_key, maxval=5e-2, shape=(batch, n_vars))
            x_dummy = jax.device_put(jnp.full((batch, n_vars), fill_value=0.99, dtype=float) - noise, batch_sharding)

        if not benchmark:
            if mesh.shape['batch'] > 1:
                print("Batch sharding:")
                jax.debug.visualize_array_sharding(x_dummy)
            if mesh.shape['objective'] > 1:
                print("Objective sharding:")
                jax.debug.visualize_array_sharding(objectives[0].clauses.lits)

        peak_mem = solver.peak_memory_estimation((x_dummy, weights))
        mem_est_per_point = peak_mem // batch
        print(mem_est_per_point)
        return()
        warm_start = time()
        solver.warmup((x_dummy, weights), counting)
        warm_end = time()
        if not counting and solver.warmup_sol:
            # Found a solution during warmup which we have printed. Exit now.
            print("W-TTFS", warm_end - warm_start)
            print("W-XT", warm_end - warm_start)
            return warm_end - warm_start

    all_sols_cnt = 0
    all_sols = None
    best_x = None
    best_unsat = jnp.inf
    best_unsat_clauses_idx = None
    first_sol = None
    starts = 0
    restart_ct = 0
    timeout_m, timeout_s = divmod(timeout, 60)
    pbar = tqdm(
            total=timeout,
            desc=f"restart {restart_ct} ({starts}/{restart} -- best={best_unsat})",
            bar_format="{l_bar}{bar}| {elapsed}/" + str(timeout_m).zfill(2) + ":" + str(timeout_s).zfill(2),
            ) if not benchmark else None
    t0 = time()
    batch_time = t0
    accum_descent = 0

    while (time() - t0 < timeout) and (not solver.warmup_sol or counting):
        tloop = time()

        # Randomisation & Init
        key, s_key = jax.random.split(key)
        f_key, s_f_key = jax.random.split(f_key)
        x0 = x0_guesses(s_key, batch, n_vars, sample_method, prefix_vectors)
        x0 = jax.device_put(x0, batch_sharding)

        # Do it.
        opt_x0, opt_unsat, opt_iters, opt_unsat_ct, eval_scores = solver.run(x0, weights)

        accum_descent += time() - tloop
        tbatch = time()

        batch_unsat_scores = jnp.sum(opt_unsat, axis=1)
        batch_best_unsat = jnp.min(batch_unsat_scores)
        batch_best_loc = jnp.argmin(batch_unsat_scores)
        batch_best_x = opt_x0[batch_best_loc]
        batch_best_unsat_clauses_idx = jnp.nonzero(opt_unsat[batch_best_loc])
        if first_sol is None and batch_best_unsat < best_unsat:
            # If first sol is set, then a solution has already been found, and we are only here because counting=True
            best_x = np.asarray(batch_best_x).copy()
            best_unsat = np.asarray(batch_best_unsat).copy()
            best_unsat_clauses_idx = np.asarray(batch_best_unsat_clauses_idx).copy()

        found_sol = False
        if batch_best_unsat == 0:
            if not counting and not benchmark:
                print("Found a solution!")
            if first_sol is None:
                print("X-TTFS", tbatch-t0)
                first_sol = np.asarray(jnp.sign(batch_best_x)).copy()
            found_sol = True

        if fuzz_limit and counting and not found_sol:
            # Knock current batch in attempt to find more solutions.
            fuzz_attempt = 0
            F_x = opt_x0[:]
            f_key, s_f_key = jax.random.split(f_key)
            fuzz = jax.random.uniform(s_f_key, minval=1e-7, maxval=1e-2, shape=x0.shape)
            fuzz_mag = 1
            while fuzz_attempt < fuzz_limit:
                fuzz_attempt += 1
                if fuzz_mag != 1:
                    fuzz_adj = jnp.sign(fuzz) * jnp.abs(fuzz) ** (1 / fuzz_mag)
                else:
                    fuzz_adj = fuzz
                #F_xC = F_x[:]

                # # Outside tracks if any point has been fuzzed outside of bounds
                # outside = xF + fuzz_adj
                # outside = jnp.abs(flip) > 1

                # Project back on to hypercube.
                F_x = jnp.clip(F_x + fuzz_adj, -1, 1)

                F_opt_x, F_opt_unsat, F_opt_iters, F_opt_unsat_ct, F_eval_scores = solver.run(F_x, weights)

                F_batch_unsat_scores = jnp.sum(opt_unsat, axis=1)
                F_batch_best_unsat = jnp.min(F_batch_unsat_scores)
                F_batch_best_loc = jnp.argmin(F_batch_unsat_scores)
                F_batch_best_x = opt_x0[F_batch_best_loc]
                F_batch_best_unsat_clauses_idx = jnp.nonzero(opt_unsat[F_batch_best_loc])
                if F_batch_best < best_unsat:
                    #TODO: We make a (spurious?) assumption that bumping a solution will find that solution again
                    # So this check needs to be adjusted to also check the number the solutions and replace only if
                    # find more - this covers both the counting and not counting case.
                    #TODO: This would also subsume the next check somewhat - since found_sol would already be true.
                    # We beat the prefuzz convergence, so keep this result instead.
                    best_x = np.asarray(F_batch_best_x).copy()
                    best_unsat = np.asarray(F_batch_best_unsat).copy()
                    best_unsat_clauses_idx = np.asarray(F_batch_best_unsat_clauses_idx).copy()
                    opt_x0, opt_unsat, opt_iters, opt_unsat_ct = F_x, F_opt_unsat, F_opt_iters, F_opt_unsat_ct

                if batch_bestF == 0:
                    print(f"Fuzz {fuzz_attempt} found a solution!")
                    found_sol = True
                    break

                if (jnp.sign(xFC) == jnp.sign(opt_xF)).all():
                    # If no points ended up changing signs after convergence, we didn't move at all. Increase magnitude
                    fuzz_mag += 1
                else:
                    fuzz_mag = 1

                xF = opt_xF

        if found_sol:
            if counting == 2:
                a = time()
                sol_locs = jnp.argwhere(jnp.where(batch_unsat_scores < 1, 1, 0)).flatten().tolist()
                all_sols_cnt += len(sol_locs)
                a = time()
                batch_sols = np.asarray(np.sign(np.asarray(opt_x0[sol_locs,:])))
                batch_sols = np.unique(batch_sols, axis=0) ## LONG PART
                if all_sols is not None:
                    all_sols = np.unique(np.concatenate((all_sols, batch_sols), axis=0), axis=0)
                else:
                    all_sols = batch_sols
                # for sol in sol_locs:
                #     all_sols.append(jnp.sign(opt_x0[sol]))
            if not counting:
                print("not counting??")
                break
        # clauses = {}
        # best_start = x_starts[loc]

        # if found_sol:
        #     print(f"Optim Iters: min: {jnp.min(opt_iters)}, max: {jnp.max(opt_iters)}, mean: {jnp.mean(opt_iters)}")
        #     #print("SAT! at index {} with starting point {} (ending {})".format(loc, best_start, best_x))
        #     print("SAT! at index {}".format(max((starts - 1), 0) * batch + loc))
        #     stamp = time()
        #     print(f"More stats for sol:\nstart:\n{loc}\niters:\n{opt_iters[loc]} eval:{eval_scores[loc]}\nunsat:\n{opt_unsat[loc]}\nended:\n{opt_x0[loc]}")
        #     break

        #print(f"Optim Iters: min: {jnp.min(opt_iters)}, max: {jnp.max(opt_iters)}, mean: {jnp.mean(opt_iters)}")
        starts += 1
        if restart and starts >= restart:
            iters = jnp.concat(all_dcnt_iters)
            unsat_ct = jnp.array(all_unsat_cts)

            # Reset/update for next restart
            restart_ct += 1
            seed = int(time()) if benchmark else restart_ct
            key = jax.random.PRNGKey(np.array(seed))
            f_key = jax.random.PRNGKey(np.array(seed + 1))
            starts = 0
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
                weight = jnp.where(w_pens > 0, weight + 0.1 * w_pens / worst, 1)
                #weights[idx] = jax.device_put(adj_weight, objective_sharding)
                pen_start += n_clause

        if pbar:
            pbar.set_description(f"restart {restart_ct} ({starts}/{restart} -- best={best_unsat})")
            pbar.update(time() - batch_time)
        batch_time = time()
        #del x0
        #jax.clear_caches()
        # print("end loop", time() - tloop)
    tsolve = time() - t0

    if pbar:
        pbar.close()

    p_sol = 0
    a = time()

    all_sols = [first_sol] if all_sols is None and first_sol is not None else all_sols
    if all_sols is not None:
        for sol in all_sols:
            out_string = "v"
            #assignment = []
            for i in range(n_vars):
                lit = i + 1
                if sol[i] > 0:
                    out_string += f" {-lit}"
                    #assignment.append(-lit)
                else:
                    out_string += f" {lit}"
                    #assignment.append(lit)
            print(out_string)
            p_sol += 1
            if benchmark:# and p_sol > 3:
                break

    print("START EXP")
    print("X-GPU", n_devices)
    print("X-PPBATCH", batch)
    print("X-PPGPUBATCH", batch//n_devices, "PPBATCH/GPU")
    if warmup:
        print("X-PEAK", peak_mem)
        print("X-PEAKPP", mem_est_per_point, "PEAK/PPBATCH")
    print("X-BATCHES", starts)
    print("X-PTOTAL", starts*batch, "BATCHES*PPBATCH")
    print("X-LOOP", tsolve)
    print("X-DESC", accum_descent)
    if all_sols is not None and first_sol is not None:
        print("X-SOLS", all_sols_cnt)
        print("X-UQSOLS", len(all_sols))
    else:
        print("X-SOLS", 0)
        print("X-UQSOLS", 0)
    print("X-RATIODESC", (starts*batch)/accum_descent, "POINTS/DESCTIME")
    print("X-RATIOLOOP", (starts*batch)/tsolve, "POINTS/LOOPTIME")
    print("END EXP")
    return tsolve


def main(
    file: str,
    timeout: int,
    batch: int | None,
    restart: int,
    fuzz: int,
    n_devices: int,
    disk_cache: str = None,
    benchmark: bool = False,
    counting: int = 0,
    warmup: bool = False,
    rand_seed: bool = False,
    q_bench: bool = False
) -> None:
    if file is None:
        print("Error: Please provide a (hybrid) dimacs CNF file")
        return 1

    sat_parser = PBSATFormula(workers=4, n_devices=n_devices, disk_cache=disk_cache, benchmark=benchmark)
    stamp1 = time()
    sat_parser.read_DIMACS(file)
    stamp2 = time()
    read_time = stamp2 - stamp1

    n_var = sat_parser.n_var
    n_clause = sat_parser.n_clause
    objectives = sat_parser.process_clauses()
    stamp1 = time()
    process_time = stamp1 - stamp2

    if q_bench:
        opt_batch = run_solver(timeout, n_var, n_clause, batch, restart, fuzz, objectives, \
                         n_devices=n_devices, counting=counting, benchmark=benchmark, warmup=warmup, q_bench=1)
        fractions_neighbourhood = [0.55, 0.66, 0.75, 0.85, 0.95]
        small = [x for x in [int(opt_batch*(1/(2**i))) for i in range(20,0,-1)] if x > 63 and x < 0.55*opt_batch]
        batch_spectrum = small + [int(opt_batch*frac) for frac in fractions_neighbourhood] + [opt_batch] \
        + [int(opt_batch/frac) for frac in fractions_neighbourhood[::-1]]
        print(batch_spectrum)
            # Memory estimate for batch tuning, then warmup precompilation.
    #for test_batch in [2**7, 2**8]+[2**8+j for j in range(50, (2**11-2**8)+1, 50)]+[2**i for i in range(12,22)]:
    # for test_batch in [2**i for i in (list(range(8,12))+list(range(18,20)))[::-1]]:
    
    #     #print("Batch Memory Consumption estimate with size:", test_batch)
        print(f"MEMORY THROUGHPUT PROFILING (OPT = {opt_batch})")
        for test_batch in list(range(645,646,1)):# + list(range(1747,1748,1)):#[244, 488, 978, 1956, 3912, 7826]:#batch_spectrum[:6]:#[::-1]:
            jax.clear_caches()
            low = 645
            high = 1747
            for r in range(1):
                tsolve = run_solver(20, n_var, n_clause, (test_batch+4)*16, restart, fuzz, objectives, \
                         n_devices=n_devices, counting=counting, benchmark=True, warmup=not r, q_bench=2)

    else:
        t_solve = run_solver(timeout, n_var, n_clause, batch, restart, fuzz, objectives, \
                         n_devices=n_devices, counting=counting, benchmark=benchmark, warmup=warmup, \
                         prefix_vectors=None)  # TODO: Add command line arg for prefix_vectors
    # print("Some stats")
    # print("Time reading input:", read_time)
    # print("Time processing to Arrays:", process_time)
    # print("Time spent solving:", t_solve)


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
    ap.add_argument("-b", "--batch", type=int, default=-1, help="Batch size. -1 computes heuristic maximum")
    ap.add_argument("-f", "--fuzz", type=int, default=0, help="Number of times to attempt fuzzing per batch")
    ap.add_argument("-r", "--restart", type=int, default=0, help="Batches before reweight and restart (never if 0)")
    ap.add_argument("-n", "--n_devices", type=int, default=n_devices, help="Devices (eg. GPUs) to use. 0 uses all")
    ap.add_argument("-e", "--benchmark", action="store_true", help="Benchmark mode (reduce output)")
    ap.add_argument("-c", "--counting", type=int, default=0, help="Counting mode. Count solns until timeout")
    ap.add_argument("-w", "--warmup", action="store_true", help="Perform a warmup run before starting timer.")
    ap.add_argument("-d", "--debug", choices=LOG_LEVELS, default="ERROR", help=f"Set logging level ({LOG_LEVELS})")
    ap.add_argument("-s", "--rand_seed", action="store_true", help=f"Randomise seed")
    ap.add_argument("-q", "--q_bench", action="store_true", help=f"Memory Bench")

    arg = ap.parse_args()
    print("Warmup:", arg.warmup, "|| Benchmark:", arg.benchmark, \
            "|| Count Sols:", arg.counting, "|| Random Seed:", arg.rand_seed, " || Batch Size:", arg.batch)
    if arg.warmup:
        jax.clear_caches()

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
    profiler = jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=False) if arg.profile else nullcontext()
    with profiler:
        main(arg.file, arg.timeout, arg.batch, arg.restart, arg.fuzz, arg.n_devices, \
            benchmark=arg.benchmark, counting=arg.counting, warmup=arg.warmup, rand_seed=arg.rand_seed,
            q_bench=arg.q_bench)
        if arg.profile:
            jax.profiler.save_device_memory_profile("memory.prof")