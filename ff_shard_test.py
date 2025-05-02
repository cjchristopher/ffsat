import os
import sys

import jax.core

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable pre-allocation
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # Use platform-specific allocator
os.environ["XLA_FLAGS"] = " ".join(
    [
        "--xla_gpu_triton_gemm_any=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true",
        "--xla_gpu_enable_highest_priority_async_stream=true",
    ]
)

import logging
from argparse import ArgumentParser as ArgParse
from collections.abc import Callable
from contextlib import nullcontext
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

jax.config.update("jax_compilation_cache_dir", "/tmp/jax-cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
#jax.config.update("jax_explain_cache_misses", True)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_enable_checks", True)
# jax.config.update("jax_disable_jit", True)


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
    print(
        f"TRACING full eval constructor ({combined}) with shapes: {[w.shape for obj in objs for w in obj.clauses if w is not None]}"
    )

    # Construct JAX sharded evaluator and verifier
    def single_eval_verify(obj: Objective) -> tuple[Callable[[Array], Array], ...]:
        print(f"TRACING single obj eval constructor with shapes: {[w.shape for w in obj.clauses if w is not None]}")
        lits = obj.clauses.lits
        sign = obj.clauses.sign
        mask = obj.clauses.mask
        cards = obj.clauses.cards
        types = obj.clauses.types
        # sparse = obj.clauses.sparse

        dft, idft = obj.ffts
        forward_mask = obj.forward_mask

        def evaluate(x: Array, weight: Array) -> Array:
            print(f"TRACING evaluate with shapes: {x.shape}, {weight.shape}")
            assignment = sign * x[lits]
            prod = jnp.prod(dft + assignment[:, None, :], axis=2, where=forward_mask)
            clause_eval = jnp.sum(idft * prod, axis=1).real
            # TODO: think about this
            cost = jnp.where(clause_eval > 0, clause_eval * weight, clause_eval)
            # cost = weight * clause_eval
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
        print(f"TRACING evaluate_all with shapes: {x.shape}, {[w.shape for w in weights]}")
        costs = [eval_fn(x, weight) for (eval_fn, weight) in zip(evaluators, weights)]
        cost = jnp.sum(jnp.array(costs))
        #jax.debug.print("xx: {}, {}, {}, {}", x.shape, x[:3], cost, costs)
        # jax.core.
        # if jax.core.cur_sublevel().name:
        #     jax.debug.print("XX LOC: {}", jax.core.cur_sublevel().name)
        return cost, cost  # aux info

    def verify_all(x: Array) -> Array:
        all_res = [res_fn(x) for res_fn in verifiers]
        res = jnp.concat(all_res)
        return res

    if not combined:
        print("Returning single vector versions")
        return evaluate_all, verify_all

    def evaluate_batch(xB: Array, weights: list[Array]) -> Array:
        print(f"TRACING evaluate_batch with shapes: {xB.shape}, {[w.shape for w in weights]}")
        # def eval_single(x):
        #     return evaluate_all(x, weights)
        # batch_eval, _ = jax.lax.map(eval_single, xB)
        batch_eval, _ = jax.vmap(evaluate_all, in_axes=(0, None))(xB, weights)
        # #jax.debug.print("Batch scores: {} | Mean: {}", batch_eval, jnp.mean(batch_eval), )
        # TODO: It's not clear that mean is the best here.
        return jnp.mean(batch_eval), batch_eval

    def verify_batch(xB: Array) -> Array:
        batch_res = jax.vmap(verify_all)(xB)
        return batch_res

    print("Returning batch versions")
    return evaluate_batch, verify_batch


# TODO: Enable solver switching based on command line.
def solver_config(solver: str, eval_fun: Callable):
    match solver:
        case "lbfgsb":
            return LBFGSB(fun=eval_fun, maxiter=500, has_aux=True)
        case "pgd":
            return ProjectedGradient(fun=eval_fun, projection=projection_box, maxiter=50, has_aux=True)
        case "prox":
            return ProximalGradient()


# TODO: Enable starting point selection strategies on command line (will subsume vertex start and fuzzing)
def next_x0_strategy():
    pass


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
) -> float:
    # Create device mesh, declare JAX specs, and shard:
    tsolver = time()
    #print("Solver Init", file=sys.stderr)
    devices = jax.devices("gpu")
    gpu_batch = batch_sz * len(devices)
    weights = [jnp.ones_like(obj.clauses.types) for obj in objs]
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
    #print("Complete Init", time() - tsolver, file=sys.stderr)
    t0 = time()

    with mesh:
        #print("Before shardings", time() - tsolver, file=sys.stderr)
        clause_sharding = NamedSharding(mesh, shard_spec)
        sharded_objs = tuple(shard_objective(obj, clause_sharding) for obj in objs)
        sharded_weight = [jax.device_put(weight, clause_sharding) for weight in weights]
        #print("After shardings", time() - tsolver, file=sys.stderr)

        def process_batch(x_batch: Array, weights: list[Array]) -> tuple[Array, Array, Array, Array]:
            print(
                f"TRACING process_batch at time {time() - t0:.2f}s with shapes: {x_batch.shape}, {[w.shape for w in weights]}"
            )
            evaluator, verifier = eval_verify(sharded_objs, opt_combined)
            # TODO: Set solver and .run args as well somehow.
            solver = solver_config("lbfgsb", evaluator)  # lbfgsb, pgd

            def optimize(x: Array) -> tuple[Array, Array, Array]:
                print(f"TRACING optimize with shapes: {x.shape}")
                # x_opt, state = solver.run(x, hyperparams_proj=(-1, 1), weights=weights) #PDG
                #jax.debug.print(
                #     "Combined?: {} | x_shard shape: {} | dtype: {} | min: {} | max: {} | mean: {}",
                #     opt_combined,
                #     x.shape,
                #     x.dtype,
                #     jnp.min(x),
                #     jnp.max(x),
                #     jnp.mean(x),
                # )
                x_opt, state = solver.run(
                    x, bounds=(-1 * jnp.ones_like(x), jnp.ones_like(x)), weights=weights
                )  # L-BFGS-B
                unsat = jnp.squeeze(verifier(x_opt))
                #jax.debug.print(
                #     "AO Batch scores: {} | Mean: {} | Steps: {}", state.aux, jnp.mean(state.aux), state.iter_num
                # )
                return x_opt, unsat, jnp.atleast_1d(state.iter_num), state.aux

            if not opt_combined:
                print("VMAP optimize")
                optimize_batch = jax.vmap(optimize, in_axes=(0,))
                # x_shard = x_batch
                x_shard = jax.lax.with_sharding_constraint(x_batch, clause_sharding)
            else:
                print("JIT optimize")
                optimize_batch = jax.jit(optimize)
                x_shard = x_batch

            print(f"TRACING process_batch finished {time() - t0:.2f}s")
            print(f"TRACING x_shard {x_shard.shape}")
            #jax.debug.print("First few values of x_shard: \n{}", x_shard[:, :3])
            if not opt_combined:
                t_res, _ = jax.vmap(evaluator, in_axes=(0, None))(x_batch, weights)
                #jax.debug.print("Initial Evals:\n{}\n{}", jnp.mean(t_res), t_res)
            else:
                t_res, t_list = evaluator(x_batch, weights)
                #jax.debug.print("Initial Evals:\n{}\n{}", t_res, t_list)

            x_opt, unsat, iters, eval_scores = optimize_batch(x_shard)
            #jax.debug.print("{}", unsat.shape)
            unsat_cl_count = jnp.sum(jnp.atleast_1d(unsat), axis=0)
            return x_opt, unsat, iters, unsat_cl_count, eval_scores

        #print("JIT the whole thing", time() - tsolver, file=sys.stderr)
        opt_batch = jax.jit(process_batch, in_shardings=(None, [clause_sharding] * len(weights)))
        #print("JITTED", time() - tsolver, file=sys.stderr)

        batch_time = t0
        pbar = tqdm(
            total=timeout,
            desc=f"restart {restart_ct} ({starts}/{restart} -- best={best_unsat})",
            bar_format="{l_bar}{bar}| {elapsed}/" + str(timeout_m).zfill(2) + ":" + str(timeout_s).zfill(2),
        )
        while time() - t0 < timeout:
            fh = logging.FileHandler(filename="jax_{}_{}.log".format(restart_ct, starts))
            # Remove any existing handlers
            # for handler in jaxlog.handlers[:]:
            #     jaxlog.removeHandler(handler)
            # jaxlog.addHandler(fh)

            tloop = time()
            #print("##### New Loop start || elapsed since call to run:", tloop - tsolver, file=sys.stderr)
            key, subkey = jax.random.split(key)
            fuzz_key, subfuzz_key = jax.random.split(fuzz_key)

            sample_method = "uniform"  # or 'bias' or 'coin' (weighted benoulli coin)

            match sample_method:
                case "bias":
                    # Generate values biased towards 1 (False) while maintaining full [-1, 1] range
                    # Method 1: Using power transformation with bias_strength (original)
                    u = jax.random.uniform(subkey, minval=0.0, maxval=1.0, shape=(gpu_batch, n_vars))
                    bias_strength = 0.5  # Lower value = stronger bias towards 1 (values 0-1)
                    x0 = 2 * u**bias_strength - 1
                case "coin":
                    # Method 2: Using a biased coin flip (Bernoulli distribution)
                    # Uncomment to use this method instead
                    false_prob = 0.7  # Y% probability of generating values tending towards False
                    coin_key, subkey = jax.random.split(subkey)
                    biased_coins = jax.random.bernoulli(coin_key, p=false_prob, shape=(gpu_batch, n_vars))
                    signs = 2 * biased_coins - 1  # Convert to -1/1
                    magnitudes = jax.random.uniform(subkey, minval=0.0, maxval=1.0, shape=(gpu_batch, n_vars))
                    x0 = signs * magnitudes
                case "uniform" | _:
                    x0 = jax.random.uniform(subkey, minval=-1, maxval=1, shape=(gpu_batch, n_vars))

            fuzz = jax.random.uniform(subfuzz_key, minval=1e-10, maxval=1e-4, shape=x0.shape)
            fuzz_attempt = 0

            if v_start:
                # snap to nearest vertex and fuzz to avoid immediate saddle
                x0 = jnp.sign(x0)
                x0 = x0 + -x0 * fuzz
            # x_starts = x0[:]
            #print("Random keys regen & snap", time() - tloop, file=sys.stderr)

            # Do it.
            #print("Before OPT call", time() - tloop, file=sys.stderr)
            # with jax.profiler.trace(f"/tmp/jax-trace-restart-{restart_ct}", create_perfetto_link=True):
            #     opt_x0, opt_unsat, opt_iters, opt_unsat_ct, eval_scores = opt_batch(x0, sharded_weight)
            print(x0.shape)
            opt_x0, opt_unsat, opt_iters, opt_unsat_ct, eval_scores = opt_batch(x0, sharded_weight)
            #print("After OPT call", time() - tloop, file=sys.stderr)
            batch_unsat_scores = jnp.sum(jnp.atleast_2d(opt_unsat), axis=1)
            print("unsat scores:", batch_unsat_scores, "\n eval scores:", eval_scores)
            batch_best = jnp.min(batch_unsat_scores)
            print("eval of best score:", batch_best, jnp.max(jnp.abs(eval_scores)))
            best_unsat = batch_best if batch_best < best_unsat else best_unsat
            #print("After some JNP + maybe fuzz next", time() - tloop, file=sys.stderr)

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
                    flip = xF + fuzz**(1/fuzz_mag)
                    flip = (jnp.abs(flip) > 1)
                    #fuzz = jnp.where(flip, -fuzz, fuzz)
                    #xF = xF + -jnp.sign(xF) * fuzz**fuzz_mag
                    fuzz = (jnp.sign(fuzz)*jnp.abs(fuzz)**(1/fuzz_mag))
                    print(fuzz[0][0:5])
                    xF = jnp.clip(xF + fuzz, -1, 1)
                    print(f"FUZZ {fuzz_mag}:", xF[0][:5])
                    #print("Before OPT+FUZZ call", time() - tloop, file=sys.stderr)
                    opt_xF, opt_unsatF, opt_itersF, opt_unsat_ctF, eval_scoresF = opt_batch(xF, sharded_weight)
                    #print("After OPT+FUZZ call", time() - tloop, file=sys.stderr)
                    batch_unsatF_scores = jnp.sum(jnp.atleast_2d(opt_unsatF), axis=1)
                    batch_bestF = jnp.min(batch_unsatF_scores)
                    #print("JNP stuff on FUZZ results", time() - tloop, file=sys.stderr)
                    if batch_bestF < best_unsat:
                        best_unsat = batch_bestF
                        opt_x0, opt_unsat, opt_iters, opt_unsat_ct = xF, opt_unsatF, opt_itersF, opt_unsat_ctF
                    if batch_bestF == 0:
                        print(f"Fuzz {fuzz_attempt} found a solution!")
                        found_sol = True
                        break
                    if (jnp.sign(xFC) == jnp.sign(opt_xF)).all():
                        print("INCREASE MAGNITUDE", file=sys.stderr)
                        fuzz_mag += 1
                    else:
                        mymask = jnp.nonzero(jnp.sign(xFC) != jnp.sign(opt_xF))
                        fnz = (mymask[0][0],mymask[1][0])
                        print(f"diff at pos {fnz}, start {xFC[fnz]} + fuzz {fuzz[fnz]} = {xF[fnz]}. Descended to {opt_xF[fnz]}")
                        fuzz_mag = 1
                    xF = opt_xF
                    #print(f"Optim Iters: min: {jnp.min(opt_itersF)}, max: {jnp.max(opt_itersF)}, mean: {jnp.mean(opt_itersF)}")
            #print(fuzz_limit, "Fuzz complete", time() - tloop, file=sys.stderr)

            loc = jnp.argmin(batch_unsat_scores)
            best_x = opt_x0[loc]
            best_unsat_clause_idx = jnp.nonzero(jnp.atleast_2d(opt_unsat)[loc])
            #print("Stats gather", time() - tloop, file=sys.stderr)
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
            starts += gpu_batch
            if starts >= restart:
                #break
                #print("Performing restart", time() - tloop, file=sys.stderr)
                #print("Memory before restart:", jax.devices()[0].memory_stats(), file=sys.stderr)
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
                    f"Unsat counts: {penalty}, Objs: {[obj.clauses.lits.shape for obj in objs]}, Best: {best_unsat_clause_idx}"
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
                tx = time()
                #print(time() - tx, "SEP time to reload weights", file=sys.stderr)

                stamp = time()
                print("Restart time: {:.2f}/{}".format(stamp - t0, timeout))
                #print("Memory after restart:", jax.devices()[0].memory_stats(), file=sys.stderr)
                #print("Restart complete", time() - tloop, file=sys.stderr)

            pbar.set_description(f"restart {restart_ct} ({starts}/{restart} -- best={best_unsat})")
            pbar.update(time() - batch_time)
            batch_time = time()
            #print("end loop", time() - tloop, file=sys.stderr)
        pbar.close()
    return time() - t0


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
    profiler = jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True) if args.profile else nullcontext()
    with profiler:
        main(args.file, args.mode, args.timeout, args.batch, args.restart, args.fuzz, args.combine, args.vertex)
