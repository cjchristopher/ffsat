# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
"""
GPU-accelerated beam search SAT solver with fused inner loop.

This version keeps the entire beam search loop on GPU, only returning to CPU
for clause reweighting and progress reporting. Uses jax.lax.while_loop for
early termination support.
"""

from __future__ import annotations

import functools
import logging
import math
import os
from argparse import ArgumentParser as ArgParse
from time import perf_counter as time
from typing import NamedTuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.95"
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
        "--xla_gpu_require_complete_aot_autotune_results=true",
    ]
)

from collections.abc import Callable
from typing import TypeAlias

import jax
import jax.experimental.host_callback
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding
from tqdm.auto import tqdm
from utils import get_gpu_l2_cache_size

from boolean_whf import ClauseArrays, clause_type_ids
from sat_loader import PBSATFormula

logger = logging.getLogger(__name__)

jax.config.update("jax_platform_name", "gpu")
jax.config.update("jax_enable_x64", False)
jax.config.update("jax_use_shardy_partitioner", True)
jax.config.update("jax_memory_fitting_level", "O3")
jax.config.update("jax_optimization_level", "O3")
jax.config.update("jax_compiler_enable_remat_pass", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

print = functools.partial(print, flush=True)

VerifyFn: TypeAlias = Callable[[Array, Array], tuple[Array, Array]]

DEFAULT_INNER_ITERS = 10


class BeamState(NamedTuple):
    """State carried through beam search iterations."""
    points: Array           # (batch_size, n_vars) current beam
    best_candidate: Array   # (n_vars,) best assignment seen
    best_unsat: Array       # () scalar, best unsat count
    clause_totals: Array    # (n_clauses,) accumulated for reweighting
    iter_count: Array       # () iteration counter
    rng_key: Array          # PRNG state
    done: Array             # () bool, early exit flag
    prefix_idx: Array       # (batch_size,) int32, maps each point to its prefix (multi-prefix)


def build_verifier(cls: tuple[ClauseArrays, ...]) -> VerifyFn:
    """Build a verification function that returns weighted scores and unsat mask."""

    def single_verifier(cl: ClauseArrays) -> Callable[[Array], Array]:
        lits = cl.lits
        sign = cl.sign
        mask = cl.mask
        cards = cl.cards
        types = cl.types
        clause_count = lits.shape[0]
        is_negated = sign < 0

        unsat_rules = {
            "xor": lambda x: jnp.sum(x, axis=-1, where=mask) % 2 == 0,
            "cnf": lambda x: ~jnp.any(x, axis=-1, where=mask),
            "eo": lambda x: jnp.sum(x, axis=-1, where=mask) != 1,
            "amo": lambda x: jnp.sum(x, axis=-1, where=mask) > 1,
            "nae": lambda x: ~(jnp.any(x, axis=-1, where=mask) & jnp.any(~x, axis=-1, where=mask)),
            "card": lambda x: jnp.where(
                cards < 0,
                jnp.sum(x, axis=-1, where=mask) >= jnp.abs(cards),
                jnp.sum(x, axis=-1, where=mask) < cards,
            ),
            "ek": lambda x: jnp.sum(x, axis=-1, where=mask) != cards,
        }

        def verify(x: Array) -> Array:
            clause_assigned = x[:, lits] ^ is_negated
            unsat = jnp.zeros((x.shape[0], clause_count), dtype=int)
            for clause_type, rule in unsat_rules.items():
                type_id = clause_type_ids[clause_type]
                unsat_clauses = rule(clause_assigned)
                unsat += jnp.where(types == type_id, unsat_clauses, 0)
                # unsat = unsat | jnp.where(types == type_id, unsat_clauses, unsat)
            return unsat

        return verify

    verifiers = [single_verifier(cl) for cl in cls]

    def combined_verify(x: Array, weights: Array) -> tuple[Array, Array]:
        all_unsat = [v(x) for v in verifiers]
        combined = jnp.concatenate(all_unsat, axis=-1)
        weighted_scores = jnp.sum(combined.astype(jnp.float32) * weights, axis=-1)
        return weighted_scores, combined

    return combined_verify


def get_mesh(devices: list) -> tuple[Mesh, NamedSharding]:
    """Create mesh and sharding for batch parallelism."""
    mesh = Mesh(np.array(devices).reshape((len(devices),)), ("batch",))
    batch_sharding = NamedSharding(mesh, jax.sharding.PartitionSpec("batch"))
    return mesh, batch_sharding


def make_gpu_inner_loop(
    n_vars: int,
    n_clauses: int,
    batch_size: int,
    n_keep: int,
    n_cull: int,
    top_m: int,
    counting: bool,
    verifier: VerifyFn,
    flip_mask: Array,
    n_flip: int,
    # Single-prefix support (all points share one prefix)
    fixed_mask_1d: Array | None = None,
    prefix_bools_1d: Array | None = None,
    # Multi-prefix support (different prefixes per point)
    all_fixed_masks: Array | None = None,
    all_prefix_bools: Array | None = None,
    n_prefix: int = 0,
):
    """Factory to create GPU inner loop with constants closed over.

    Prefix modes:
      - No prefix:      flip_mask is (n_vars, n_vars), n_flip == n_vars.
      - Single prefix:  flip_mask is (n_free, n_vars), n_flip == n_free.
                         fixed_mask_1d / prefix_bools_1d are (n_vars,).
      - Multi-prefix:   flip_mask is (n_vars, n_vars), n_flip == n_vars.
                         Penalty applied per-point via all_fixed_masks lookup.
                         prefix_idx tracked through BeamState for selection.
    """

    n_expanded = batch_size * top_m
    single_prefix = fixed_mask_1d is not None
    multi_prefix = all_fixed_masks is not None
    BIG_PENALTY = 1e10

    # Host-side solution collection
    collected_solutions: list[np.ndarray] = []

    def host_collect_solutions(potential_sols: np.ndarray, sol_indices: np.ndarray) -> None:
        """Host callback - extracts actual solutions using sentinel-based filtering."""
        for i, idx in enumerate(sol_indices):
            if idx >= 0:
                collected_solutions.append(potential_sols[i].copy())

    def get_solutions() -> list[np.ndarray]:
        return collected_solutions

    def clear_solutions() -> None:
        collected_solutions.clear()

    def beam_expand(
        points: Array, weights: Array, rng_key: Array,
        point_fixed_masks: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """Expand all points to their best neighbors.

        Args:
            point_fixed_masks: (batch_size, n_vars) bool penalty mask for multi-prefix.
                               None for no-prefix and single-prefix modes.
        """
        keys = jax.random.split(rng_key, batch_size)

        if multi_prefix and point_fixed_masks is not None:
            def single_point_step_mp(
                point: Array, point_mask: Array, key: Array,
            ) -> tuple[Array, Array, Array]:
                candidates = point ^ flip_mask  # (n_vars, n_vars)
                weighted_scores, unsat_masks = verifier(candidates, weights)
                unsat_counts = jnp.sum(unsat_masks, axis=-1)
                noise = jax.random.uniform(key, shape=(n_flip,), minval=0, maxval=0.5)
                noisy_scores = weighted_scores + noise
                # Penalise flipping any fixed variable for this point.
                noisy_scores = noisy_scores + point_mask * BIG_PENALTY

                if top_m > 1:
                    top_idx = jnp.argsort(noisy_scores)[:top_m]
                    return candidates[top_idx], unsat_counts[top_idx], unsat_masks[top_idx]
                else:
                    best = jnp.argmin(noisy_scores)
                    return candidates[best], unsat_counts[best], unsat_masks[best]

            neighbors, unsats, masks = jax.vmap(single_point_step_mp)(
                points, point_fixed_masks, keys,
            )
        else:
            def single_point_step(point: Array, key: Array) -> tuple[Array, Array, Array]:  # type: ignore[misc]
                candidates = point ^ flip_mask  # (n_flip, n_vars)
                weighted_scores, unsat_masks = verifier(candidates, weights)
                unsat_counts = jnp.sum(unsat_masks, axis=-1)
                #jax.debug.print("{}, {}", weighted_scores.shape, unsat_counts.shape)
                noise = jax.random.uniform(key, shape=(n_flip,), minval=0, maxval=0.5)
                noisy_scores = weighted_scores + noise

                if top_m > 1:
                    top_idx = jnp.argsort(noisy_scores)[:top_m]
                    return candidates[top_idx], unsat_counts[top_idx], unsat_masks[top_idx]
                else:
                    best = jnp.argmin(noisy_scores)
                    return candidates[best], unsat_counts[best], unsat_masks[best]

            neighbors, unsats, masks = jax.vmap(single_point_step)(points, keys)

        return (
            neighbors.reshape(-1, n_vars),
            unsats.reshape(-1),
            masks.reshape(-1, n_clauses),
        )

    def select_and_refill(
        expanded: Array, unsat_masks: Array, weights: Array, fill_key: Array,
        expanded_prefix_idx: Array | None = None,
    ) -> tuple[Array, Array]:
        """Select top candidates, cull worst and refill with random if beta > 0.

        Returns:
            (new_points, new_prefix_idx)
        """
        scores = jnp.sum(unsat_masks.astype(jnp.float32) * weights, axis=-1)
        sorted_idx = jnp.argsort(scores)
        kept = expanded[sorted_idx[:n_keep]]

        # Track prefix assignment through selection
        if expanded_prefix_idx is not None:
            kept_prefix_idx = expanded_prefix_idx[sorted_idx[:n_keep]]
        else:
            kept_prefix_idx = jnp.zeros(n_keep, dtype=jnp.int32)

        if n_cull > 0:
            random_fill = jax.random.bernoulli(fill_key, p=0.5, shape=(n_cull, n_vars))

            # Apply prefix values to random fills
            if single_prefix:
                assert fixed_mask_1d is not None and prefix_bools_1d is not None
                random_fill = jnp.where(fixed_mask_1d, prefix_bools_1d, random_fill)
                fill_prefix_idx = jnp.zeros(n_cull, dtype=jnp.int32)
            elif multi_prefix:
                assert all_fixed_masks is not None and all_prefix_bools is not None
                # Round-robin prefix assignment for refills
                fill_prefix_idx = jnp.arange(n_cull, dtype=jnp.int32) % n_prefix
                fill_masks = all_fixed_masks[fill_prefix_idx]    # (n_cull, n_vars)
                fill_values = all_prefix_bools[fill_prefix_idx]  # (n_cull, n_vars)
                random_fill = jnp.where(fill_masks, fill_values, random_fill)
            else:
                fill_prefix_idx = jnp.zeros(n_cull, dtype=jnp.int32)

            new_points = jnp.concatenate([kept, random_fill], axis=0)
            new_prefix_idx = jnp.concatenate([kept_prefix_idx, fill_prefix_idx], axis=0)
            return new_points, new_prefix_idx

        return kept, kept_prefix_idx

    def make_inner_loop(weights: Array, n_iters: int):
        """Create JIT-compiled inner loop for given weights and iteration count."""

        def body_fn(state: BeamState) -> BeamState:
            rng_key, step_key, fill_key = jax.random.split(state.rng_key, 3)

            # Build per-point fixed masks for multi-prefix penalty
            point_fixed_masks = None
            if multi_prefix:
                assert all_fixed_masks is not None
                point_fixed_masks = all_fixed_masks[state.prefix_idx]  # (batch, n_vars)

            # Expand
            expanded, unsats, unsat_masks = beam_expand(
                state.points, weights, step_key, point_fixed_masks,
            )

            # Accumulate for reweighting
            new_clause_totals = state.clause_totals + unsat_masks.sum(axis=0).astype(jnp.float32)

            # Find best this iteration
            best_idx = jnp.argmin(unsats)
            iter_best = expanded[best_idx]
            iter_best_unsat = unsats[best_idx]
            #jax.debug.print("{}, {}, {}", best_idx, iter_best, iter_best_unsat)

            # Update global best
            is_better = iter_best_unsat < state.best_unsat
            new_best_candidate = jnp.where(is_better, iter_best, state.best_candidate)
            new_best_unsat = jnp.minimum(iter_best_unsat, state.best_unsat)

            # Track prefix_idx through expansion (each point produces top_m children)
            expanded_prefix_idx = jnp.repeat(state.prefix_idx, top_m) if multi_prefix else None

            # Selection
            new_points, new_prefix_idx = select_and_refill(
                expanded, unsat_masks, weights, fill_key, expanded_prefix_idx,
            )

            # Handle solutions
            sol_mask = unsats == 0
            n_sols = sol_mask.sum()

            if counting:
                # Stream all solutions to host via callback (use -1 sentinel for invalid)
                sol_indices = jnp.where(sol_mask, size=n_expanded, fill_value=-1)[0]
                potential_sols = expanded[jnp.maximum(sol_indices, 0)]  # Safe indexing
                jax.experimental.io_callback(
                    host_collect_solutions,
                    (),
                    potential_sols,
                    sol_indices,
                    ordered=False,
                )
                new_done = jnp.array(False)
            else:
                # Exit on first solution
                new_done = n_sols > 0

            return BeamState(
                points=new_points,
                best_candidate=new_best_candidate,
                best_unsat=new_best_unsat,
                clause_totals=new_clause_totals,
                iter_count=state.iter_count + 1,
                rng_key=rng_key,
                done=new_done,
                prefix_idx=new_prefix_idx,
            )

        def cond_fn(state: BeamState) -> Array:
            return (~state.done) & (state.iter_count < n_iters)

        @jax.jit
        def gpu_inner_loop(init_state: BeamState) -> BeamState:
            return jax.lax.while_loop(cond_fn, body_fn, init_state)

        return gpu_inner_loop

    return make_inner_loop, get_solutions, clear_solutions


def run_beam_search(
    timeout: int,
    n_vars: int,
    n_clauses: int,
    batch_size: int,
    cls: tuple[ClauseArrays, ...],
    n_devices: int = 1,
    max_iters: int = 0,
    rand_seed: bool = False,
    counting: bool = False,
    top_m: int = 1,
    beta: float = 0.0,
    restart_thresh: int = 0,
    weight_decay: float = 0.9,
    prefixes: np.ndarray | None = None,
) -> float:
    """Run parallel beam search SAT solver with fused GPU inner loop."""

    # Initialize devices and mesh
    devices = jax.devices("gpu")[:n_devices]
    mesh, batch_sharding = get_mesh(devices)
    jax.sharding.set_mesh(mesh)

    seed = int(time()) if rand_seed else 42
    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_key = jax.random.split(rng_key)

    verifier = jax.jit(build_verifier(cls))
    weights = jnp.ones(n_clauses, dtype=jnp.float32)

    # ── Prefix handling ──────────────────────────────────────────────────
    n_prefix = prefixes.shape[0] if prefixes is not None else 0
    single_prefix = (n_prefix == 1)
    multi_prefix = (n_prefix > 1)

    if single_prefix:
        assert prefixes is not None
        # Reduced flip_mask: only rows for free (unfixed) variables.
        free_indices = np.where(prefixes[0] == 0)[0]
        n_flip = len(free_indices)
        flip_mask = jnp.eye(n_vars, dtype=bool)[free_indices]  # (n_free, n_vars)
        fixed_mask_1d = jnp.array(prefixes[0] != 0)             # (n_vars,) True where fixed
        prefix_bools_1d = jnp.array(prefixes[0] < 0)            # (n_vars,) True where var=True
        print(f"Single prefix: {n_flip} free vars (reduced from {n_vars})")
    elif multi_prefix:
        assert prefixes is not None
        # Full flip_mask; penalty applied per-point in beam_expand.
        flip_mask = jnp.eye(n_vars, dtype=bool)
        n_flip = n_vars
        all_fixed_masks = jnp.array(prefixes != 0)   # (n_prefix, n_vars)
        all_prefix_bools = jnp.array(prefixes < 0)   # (n_prefix, n_vars)
        print(f"Multi-prefix: {n_prefix} prefix vectors")
    else:
        flip_mask = jnp.eye(n_vars, dtype=bool)
        n_flip = n_vars
    # ─────────────────────────────────────────────────────────────────────

    n_cull = int(batch_size * beta)
    n_keep = batch_size - n_cull

    # Determine inner loop size
    inner_iters = restart_thresh if restart_thresh > 0 else DEFAULT_INNER_ITERS

    # Adjust max_iters to multiple of inner_iters
    if max_iters > 0 and max_iters % inner_iters != 0:
        old = max_iters
        max_iters = ((max_iters // inner_iters) + 1) * inner_iters
        print(f"WARNING: Adjusted max_iters {old} -> {max_iters} (multiple of {inner_iters})")

    # Auto-select batch size
    if batch_size == -1:
        print("Guessing optimal batch size")
        l2_cache_size = get_gpu_l2_cache_size(devices[0])
        if l2_cache_size is not None:
            gpu_mem_target = int(l2_cache_size * 0.90) * n_devices * 2
            print(f"Targeting L2 cache: {l2_cache_size / (1024*1024):.1f} MB per GPU")
        else:
            gpu_mem_target = devices[0].memory_stats()["bytes_limit"] * 0.01
            print("L2 cache size unknown, using 1% VRAM heuristic")

        dtype_sz = jnp.dtype(cls[0].lits.dtype).itemsize
        all_obj_sz = sum([np.prod([*ca.lits.shape, dtype_sz]) for ca in cls])
        flip_sz = (n_clauses * n_flip + n_flip * n_vars) * jnp.dtype(bool).itemsize
        batch_size = int(np.floor((gpu_mem_target - all_obj_sz) / flip_sz)) * n_devices
        print(f"Batch size: {batch_size} (n_flip={n_flip}, consumed by clauses: {all_obj_sz / (1024*1024):.1f} MB)")

        # Recompute n_cull/n_keep with actual batch_size
        n_cull = int(batch_size * beta)
        n_keep = batch_size - n_cull

    # Batch alignment: must be divisible by n_devices and n_prefix (if any).
    alignment = math.lcm(max(n_prefix, 1), n_devices)
    batch_size = (batch_size // alignment) * alignment
    batch_size = max(batch_size, alignment)

    # Recompute n_cull/n_keep after final batch_size is settled.
    n_cull = int(batch_size * beta)
    n_keep = batch_size - n_cull

    # Create GPU loop factory
    make_inner_loop, get_solutions, clear_solutions = make_gpu_inner_loop(
        n_vars=n_vars,
        n_clauses=n_clauses,
        batch_size=batch_size,
        n_keep=n_keep,
        n_cull=n_cull,
        top_m=top_m,
        counting=counting,
        verifier=verifier,
        flip_mask=flip_mask,
        n_flip=n_flip,
        # Single-prefix constants (None when unused)
        fixed_mask_1d=fixed_mask_1d if single_prefix else None,
        prefix_bools_1d=prefix_bools_1d if single_prefix else None,
        # Multi-prefix tables (None when unused)
        all_fixed_masks=all_fixed_masks if multi_prefix else None,
        all_prefix_bools=all_prefix_bools if multi_prefix else None,
        n_prefix=n_prefix,
    )

    # Initialize points
    points = jax.random.bernoulli(init_key, p=0.5, shape=(batch_size * top_m, n_vars))

    # Apply prefix values to initial points
    if single_prefix:
        points = jnp.where(fixed_mask_1d, prefix_bools_1d, points)
    elif multi_prefix:
        # Replicate prefix bools across batch: each prefix gets (batch*top_m)/n_prefix copies
        rep = (batch_size * top_m) // n_prefix
        replicated_mask = jnp.repeat(all_fixed_masks, rep, axis=0)
        replicated_bools = jnp.repeat(all_prefix_bools, rep, axis=0)
        points = jnp.where(replicated_mask, replicated_bools, points)
        # Track which prefix each expanded point belongs to
        expanded_pidx = jnp.repeat(jnp.arange(n_prefix, dtype=jnp.int32), rep)

    if top_m > 1:
        weighted_scores, _ = verifier(points, weights)
        top_indices = jnp.argsort(weighted_scores)[:batch_size]
        points = points[top_indices]
        if multi_prefix:
            expanded_pidx = expanded_pidx[top_indices]
    points = jax.device_put(points, batch_sharding)

    # Build initial prefix_idx for multi-prefix tracking through selection
    if multi_prefix:
        if top_m > 1:
            init_prefix_idx = expanded_pidx
        else:
            init_prefix_idx = jnp.repeat(jnp.arange(n_prefix, dtype=jnp.int32), batch_size // n_prefix)
        init_prefix_idx = jax.device_put(init_prefix_idx, batch_sharding)

    # Initial state
    state = BeamState(
        points=points,
        best_candidate=points[0],
        best_unsat=jnp.array(n_clauses, dtype=jnp.int32),
        clause_totals=jnp.zeros(n_clauses, dtype=jnp.float32),
        iter_count=jnp.array(0, dtype=jnp.int32),
        rng_key=rng_key,
        done=jnp.array(False),
        prefix_idx=init_prefix_idx if multi_prefix else jnp.zeros(batch_size, dtype=jnp.int32),
    )

    total_iters = 0
    restart_ct = 0
    best_assignment = None

    pbar = tqdm(
        total=timeout,
        desc="iter 0 (best=undef)",
        bar_format="{l_bar}{bar}|{elapsed}/{total_fmt} {postfix}",
    )

    t0 = time()
    last_update = t0

    # Compile inner loop for current weights
    gpu_loop = make_inner_loop(weights, inner_iters)

    while (time() - t0 < timeout):
        if max_iters > 0 and total_iters >= max_iters:
            break

        # Reset per-batch state (keep points and best)
        state = state._replace(
            clause_totals=jnp.zeros(n_clauses, dtype=jnp.float32),
            iter_count=jnp.array(0, dtype=jnp.int32),
            done=jnp.array(False),
        )

        # ===== GPU INNER LOOP =====
        state = gpu_loop(state)
        jax.block_until_ready(state)

        total_iters += inner_iters

        # Track best assignment
        best_unsat_val = int(state.best_unsat)
        if best_assignment is None or best_unsat_val < int(np.sum(best_assignment is None or True)):
            best_assignment = np.array(state.best_candidate)

        # Check for early exit (non-counting mode)
        if state.done and not counting:
            best_assignment = np.array(state.best_candidate)
            print(f"\nSAT! Found at ~iteration {total_iters}")
            break

        # Report solutions found (counting mode)
        n_found = len(get_solutions())
        if counting and n_found > 0:
            print(f"\nSolutions so far: {n_found}")

        # Reweight clauses
        if restart_thresh > 0:
            clause_totals = np.array(state.clause_totals)
            worst = max(clause_totals.max(), 1.0)
            weights = weight_decay * weights + (1 - weight_decay) * clause_totals / worst
            restart_ct += 1
            # Recompile with new weights
            gpu_loop = make_inner_loop(weights, inner_iters)

        # Progress update
        now = time()
        if now - last_update > 0.5:
            elapsed = now - t0
            pbar.n = min(elapsed, timeout)
            pbstr = f"{total_iters % inner_iters}/{inner_iters}" if restart_thresh else f"{total_iters}"
            pbar.set_description(f"restart {restart_ct} ({pbstr} -- best={best_unsat_val})")
            pbar.set_postfix_str(f"{total_iters / elapsed:.1f} it/s")
            pbar.refresh()
            last_update = now

    pbar.close()
    solve_time = time() - t0

    # Final output
    best_unsat_val = int(state.best_unsat)
    if counting:
        all_solutions = get_solutions()
        if all_solutions:
            print("SAT!")
            assignment = all_solutions[0]
            out_string = "v"
            for i in range(n_vars):
                lit = i + 1
                out_string += f" {lit}" if assignment[i] else f" {-lit}"
            print(out_string)
            print(f"Found {len(all_solutions)} solutions")
        else:
            print(f"No solution found. Best unsat count: {best_unsat_val}")
    elif best_unsat_val == 0 and best_assignment is not None:
        print("SAT!")
        out_string = "v"
        for i in range(n_vars):
            lit = i + 1
            out_string += f" {lit}" if best_assignment[i] else f" {-lit}"
        print(out_string)
    else:
        print(f"No solution found. Best unsat count: {best_unsat_val}")
        if best_assignment is not None:
            out_string = "v"
            for i in range(n_vars):
                lit = i + 1
                out_string += f" {lit}" if best_assignment[i] else f" {-lit}"
            print(f"Best assignment: {out_string}")

    logger.info(f"X-ITERS {total_iters}")
    logger.info(f"X-TIME {solve_time}")
    logger.info(f"X-BEAM {batch_size}")
    logger.info(f"X-BEST_UNSAT {best_unsat_val}")

    return solve_time


# =============================================================================
# Entry Point
# =============================================================================


def main(
    file: str,
    timeout: int,
    batch_size: int = -1,
    n_devices: int = 1,
    disk_cache: str = "",
    counting: bool = False,
    rand_seed: bool = False,
    max_iters: int = 0,
    top_m: int = 1,
    beta: float = 0.0,
    restart_thresh: int = 0,
    weight_decay: float = 0.9,
    prefix_file: str = "",
) -> None:
    """Main entry point."""
    stamp1 = time()
    sat_parser = PBSATFormula(workers=4, n_devices=n_devices, disk_cache=disk_cache, file=file)
    stamp2 = time()
    read_time = stamp2 - stamp1

    cls = tuple(obj.clauses for obj in sat_parser.process_clauses_to_array())
    n_var = sat_parser.n_var
    n_clause = sat_parser.n_clause
    stamp1 = time()
    process_time = stamp1 - stamp2

    # Process prefixes (includes unit literals from DIMACS parsing).
    prefixes = None
    if prefix_file or sat_parser.unit_prefix:
        prefixes = sat_parser.process_prefix(prefix_file)
        n_prefix = prefixes.shape[0]
        n_fixed = np.count_nonzero(prefixes[0]) if n_prefix == 1 else [np.count_nonzero(p) for p in prefixes]
        print(f"Prefix: {n_prefix} vector(s), fixed vars: {n_fixed} / {n_var}")

    t_solve = run_beam_search(
        timeout,
        n_var,
        n_clause,
        batch_size,
        cls,
        n_devices=n_devices,
        max_iters=max_iters,
        rand_seed=rand_seed,
        counting=counting,
        top_m=top_m,
        beta=beta,
        restart_thresh=restart_thresh,
        weight_decay=weight_decay,
        prefixes=prefixes,
    )

    logger.info(f"Time reading input: {read_time}")
    logger.info(f"Time processing to Arrays: {process_time}")
    logger.info(f"Time spent solving: {t_solve}")


if __name__ == "__main__":
    n_devices = len(jax.devices("gpu"))
    print(jax.devices("gpu"))

    parser = ArgParse(description="GPU Beam Search SAT Solver (Fused)")
    parser.add_argument("file", type=str, help="Input file (.cnf, .hybrid, .opb)")
    parser.add_argument("-t", "--timeout", type=int, default=300, help="Timeout in seconds")
    parser.add_argument("-b", "--beam", type=int, default=-1, help="Beam width (-1 for auto)")
    parser.add_argument("-g", "--gpus", type=int, default=n_devices, help="Number of GPUs")
    parser.add_argument("-i", "--iters", type=int, default=0, help="Max iterations (0=unlimited)")
    parser.add_argument("-c", "--counting", action="store_true", help="Count solutions mode")
    parser.add_argument("-s", "--seed", action="store_true", help="Random seed from time")
    parser.add_argument("--cache", type=str, default="", help="Disk cache for FFT matrices")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("-m", "--top-m", type=int, default=1, help="Top m neighbors per point")
    parser.add_argument("--beta", type=float, default=0.0, help="Cull fraction (0.0-1.0)")
    parser.add_argument("-r", "--restart", type=int, default=0, help="Reweight interval (0=never)")
    parser.add_argument("-a", "--alpha", type=float, default=0.9, help="Weight decay (0.0-1.0)")
    parser.add_argument("-p", "--prefix", type=str, default="", help="Prefix file (fixed variable assignments)")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    main(
        file=args.file,
        timeout=args.timeout,
        batch_size=args.beam,
        n_devices=args.gpus,
        disk_cache=args.cache,
        counting=args.counting,
        rand_seed=args.seed,
        max_iters=args.iters,
        top_m=args.top_m,
        beta=args.beta,
        restart_thresh=args.restart,
        weight_decay=args.alpha,
        prefix_file=args.prefix,
    )
