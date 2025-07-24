from __future__ import annotations

import abc
import functools
from collections.abc import Callable
from time import perf_counter as time
from typing import Optional as Opt
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import NamedSharding
from jaxopt import LBFGSB, ProjectedGradient, ProximalGradient, ScipyBoundedMinimize
from jaxopt._src import base as job
from jaxopt.projection import projection_box as box
from scipy.optimize import Bounds, OptimizeResult
from scipy.optimize import minimize as ScipyMinimize

# import optimistix as optx
# from optimistix import BoundedLBFGS
from boolean_whf import Objective, clause_type_ids

# TODO: Import specific solver modules when needed
# from . import hj_mad
# from . import langevin_annealing
# from . import pgd
print = functools.partial(print, flush=True)

Evaluator: TypeAlias = Callable[[Array, list[Array]], Array]
Verifier: TypeAlias = Callable[[Array], Array]


def build_eval_verify(objs: tuple[Objective, ...]) -> tuple[tuple[Evaluator, ...], tuple[Verifier, ...]]:
    """
    Constructs JAX-based evaluators and verifiers for a set of objectives.
    This function generates callable evaluators and verifiers for the given objectives by closing over their constants.
    Evaluators compute the cost of assignments and verifiers check the satisfaction of clauses based on the assignment.

    Uses the convolution theorem to compute many ESP evaluations in parallel via the fourier domain.
    O(n^2) before parallelisation - tree based convolution is O(nlog^2n), but is ordered thus can't be parallelised.

    Args:
        objs (tuple[Objective, ...]): The objectives to generate evaluators and verifiers for.

    Returns:
        tuple: A tuple containing two tuples of Callables:
            - evaluators (tuple[Evaluator, ...]): Given assignment and weights, evaluate the cost of an objective 
            - verifiers (tuple[Verifier, ...]): Given assignment, check satisfaction of an objective.
    """
    def single_eval_verify(obj: Objective) -> tuple[Evaluator, Verifier]:
        lits = obj.clauses.lits
        sign = obj.clauses.sign
        mask = obj.clauses.mask
        cards = obj.clauses.cards
        types = obj.clauses.types

        dft, idft = obj.ffts
        forward_mask = True if jnp.all(obj.forward_mask) else obj.forward_mask
        clause_count = obj.clauses.lits.shape[0]

        # @jax.profiler.annotate_function
        def evaluate(x: Array, weight: Array) -> Array:
            # fmt: off
            @jax.checkpoint
            def fourier_checkpoint(assignment: Array) -> Array:
                # Add dimension to capture K+1 shifted roots for K terms of the clause.
                fourier_domain = dft + assignment[:, None, :]                     # (N?,(K+1),1) + (N,_,K) = (N,(K+1),K)
                esp_freq = jnp.prod(fourier_domain, axis=-1, where=forward_mask)
                return esp_freq                                                   # (N,(K+1))

            assignment = sign * x[lits]                                           # (N,K) * (N,K) = (N,K)
            esp_freq = fourier_checkpoint(assignment)                             # (N,(K+1))
            esp_eval = idft * esp_freq                                            # (1,(K+1)) * (N,(K+1)) = (N,(K+1))
            clause_eval = jnp.sum(esp_eval.real, axis=-1)                         # (N,)
            weighted_eval = weight * clause_eval                                  # (N,) * (N,) = (N,)
            x_eval = jnp.sum(weighted_eval, axis=-1)                              # (1,)
            #print("Executing from python")
            return jnp.atleast_1d(x_eval)
            # fmt: on

        def verify(x: Array) -> Array:
            assign = sign * x[lits]
            unsat = jnp.zeros(clause_count, dtype=bool)
            # fmt: off
            clause_unsat_rules = {
                "xor": lambda: jnp.sum(assign < 0, axis=1, where=mask) % 2 == 0,
                "cnf": lambda: jnp.min(assign, axis=1, where=mask, initial=jnp.inf) > 0,
                "eo":  lambda: jnp.sum(assign < 0, axis=1, where=mask) != 1,
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
                type_id = clause_type_ids[clause_type]
                unsat_cond = handler()
                unsat = unsat | jnp.where(types == type_id, unsat_cond, unsat)

            return unsat
            # fmt: on

        return evaluate, verify

    evaluators: tuple[Evaluator]
    verifiers: tuple[Verifier]
    evaluators, verifiers = zip(*[single_eval_verify(obj) for obj in objs])
    return evaluators, verifiers


def seq_eval_verify(evaluators: tuple[Evaluator], verifiers: tuple[Verifier]) -> tuple[Evaluator, Verifier]:
    """
    Groups a collection (usually all) of Evaluators & Verifiers into a sequence.
    """

    def seq_evals(x: Array, weights: tuple[Array]) -> tuple[Array, Array]:
        costs = [evaluate(x, weight) for (evaluate, weight) in zip(evaluators, weights)]
        cost = jnp.sum(jnp.array(costs))
        return cost, cost  # aux info - remove when consolidating

    def seq_verifies(x: Array) -> Array:
        all_res = [verify(x) for verify in verifiers]
        res = jnp.concat(all_res, axis=-1)
        return res

    return seq_evals, seq_verifies


class FFSatSolver(abc.ABC):
    def __init__(
        self,
        evaluator: Evaluator,
        verifier: Verifier,
        sol_name: str = "lbfgsb",
        shard_spec: Opt[tuple[NamedSharding, tuple[NamedSharding, ...]]] = None,
        warmup_data: tuple[Array, list[Array]] = None,
    ):
        self.sol_name = sol_name
        self.maxiter = 500
        # if self.sol_name in ["optim"]:
        #     self.solver = BoundedBFGS()
        #     self.solver = optimistix.minimise(evaluator, BoundedBFGS, args={'weights': weights}, has_aux=True)

        if self.sol_name in ["lbfgsb", "pgd", "josp-lbfgsb"]:  # probably change to if sol_name in JAX_OPTIMS
            if self.sol_name in ["lbfgsb"]:
                self.solver = LBFGSB(fun=evaluator, maxiter=self.maxiter, has_aux=True)  # , jit=False)
                print("Setting up JAXOPT L-BFGS-B:", type(self.solver))

                def opt(x: Array, weights: list[Array]) -> tuple[Array, Array, Array, Array]:
                    bounds = (-1 * jnp.ones_like(x), jnp.ones_like(x))
                    x_opt, state = self.solver.run(x, weights=weights, bounds=bounds)
                    unsat = jnp.squeeze(verifier(x_opt))
                    return x_opt, unsat, jnp.atleast_1d(state.iter_num), state.aux

            elif self.sol_name in ["josp-lbfgsb"]:
                self.solver = ScipyBoundedMinimize(fun=evaluator, method="L-BFGS-B", maxiter=self.maxiter, has_aux=True)
                print("Setting up JAXOPT ScipyBounded L-BFGS-B")

                def opt(x: Array, weights: list[Array]) -> tuple[Array, Array, Array, Array]:
                    bounds = (-1 * jnp.ones_like(x), jnp.ones_like(x))
                    x_opt, state = self.solver.run(x, weights=weights, bounds=bounds)
                    unsat = jnp.squeeze(verifier(x_opt))
                    return x_opt, unsat, jnp.atleast_1d(state.iter_num), state.fun_val

            elif self.sol_name in ["pgd"]:
                self.solver = ProjectedGradient(fun=evaluator, projection=box, maxiter=self.maxiter, has_aux=True)
                print("Setting up JAXOPT Projected Gradient (Box)")

                def opt(x: Array, weights: list[Array]) -> tuple[Array, Array, Array, Array]:
                    x_opt, state = self.solver.run(x, weights=weights, hyperparams_proj=(-1, 1))
                    unsat = jnp.squeeze(verifier(x_opt))
                    return x_opt, unsat, jnp.atleast_1d(state.iter_num), state.aux

            else:
                pass

            # def opt_ver(x: Array, weights: list[Array]) -> tuple[Array, Array, Array, Array]:
            #     x_opt, state = opt(x, weights)
            #     unsat = verifier(x_opt)
            #     unsat = jnp.squeeze(unsat)
            #     return x_opt, unsat, jnp.atleast_1d(state.iter_num), state.aux

            def vectorise(xs: Array, weights: tuple[Array]) -> tuple[Array, Array, Array, Array, Array]:
                x_opt, unsat, iters, evals = jax.vmap(opt, in_axes=(0, None))(xs, weights)
                unsat_cl_count = jnp.sum(jnp.atleast_1d(unsat), axis=0)
                return x_opt, unsat, iters, unsat_cl_count, evals

            # if shard_spec:
            #     print("JIT-Sharding Optimiser Loop")
            #     self.run = jax.jit(vectorise, in_shardings=shard_spec)
            # else:
            #     print("JIT Optimisation Loop")
            #     self.run = jax.jit(vectorise)
            self.opt = opt
            self.run = jax.jit(vectorise)

        elif self.sol_name in ["prox"]:
            self.solver = ProximalGradient
            raise NotImplementedError("HJ Moreau Proximal Gradient not yet Implemented")

        elif self.sol_name in ["langevin"]:
            # self.solver = ProximalGradient(fun=evaluator, prox=hj_moreau, maxiter=self.maxiter)
            raise NotImplementedError("Langevin Annealing not yet Implemented")

        elif self.sol_name in ["sp-lbfgsb"]:
            # We can only sensibly run this in combined multi-start since sp.minimize is max(CPU-thread) bound.
            # Here we manually VMAP and JIT the opt and ver steps
            self.solver = ScipyMinimize
            print("Setting up ScipyBounded L-BFGS-B (CPU) with JIT'd eval+verify (GPU)")
            _, _, eval_fun = job._make_funs_without_aux(fun=evaluator, value_and_grad=False, has_aux=True)
            eval_fun = jax.vmap(eval_fun, in_axes=(0, None))
            verifier = jax.vmap(verifier, in_axes=(0))
            if shard_spec:
                eval_fun = jax.jit(eval_fun, in_shardings=shard_spec)
                verifier = jax.jit(verifier, in_shardings=shard_spec[0])
            else:
                eval_fun = jax.jit(eval_fun)
                verifier = jax.jit(verifier)
            self.eval_fun = eval_fun

            def opt_ver_vectorise(x: Array, weights: list[Array]) -> tuple[Array, Array, Array, Array]:
                def flat(x0: np.ndarray) -> float:
                    # Convert back to JAX arrays, run on GPU, return CPU results to minimize.
                    x0 = jnp.array(x0).reshape(x.shape).to_device(x.sharding)
                    v, g = self.eval_fun(x0, weights)  # this is eval.
                    return float(jnp.sum(v)), np.array(g.flatten())

                x0 = np.array(x.flatten())
                bounds = Bounds(-1, 1, True)
                options = {"maxiter": self.maxiter}
                res: OptimizeResult = ScipyMinimize(flat, x0, bounds=bounds, jac=True, options=options)
                x_opt = jnp.array(res.x).reshape(x.shape)
                if shard_spec:
                    unsat = jnp.squeeze(verifier(jax.device_put(x_opt, x.sharding)))
                else:
                    unsat = jnp.squeeze(verifier(x_opt))
                unsat_cl_count = jnp.sum(jnp.atleast_1d(unsat), axis=0)
                return x_opt, unsat, jnp.array([res.nit]), unsat_cl_count, jnp.array(res.aux)

            self.run = opt_ver_vectorise

        else:
            pass

        runner = self.run if self.sol_name not in ["sp-lbfgsb"] else self.eval_fun
        print("solver.run is: (", self.sol_name, ")", self.solver, "\n runner:", runner)
        if warmup_data:
            x0, weights = warmup_data
            print("Warmup Run (Dummy Data Compilation)")
            t0 = time()
            res = runner(x0, weights)
            print([type(r) for r in res])
            print("Warmup Complete", time() - t0)
            if self.sol_name not in ["sp-lbfgsb"]:
                print("Memory Analysis:")
                memory_analysis = self.run.lower(x0, weights).compile().memory_analysis()
                print(
                    f"Total size device = {memory_analysis.temp_size_in_bytes / 1024**3} GB, "
                    f"xs+weights = {memory_analysis.argument_size_in_bytes / 1024**2} MB, "
                    f"tot: {(memory_analysis.argument_size_in_bytes + memory_analysis.temp_size_in_bytes) / 1024**3} GB"
                )


### Code for scaled log-sum-exp version of evaluate. Much more numerically stable(?), but much slower.
# # Compute constants for scaling & log-sum-exp
# mask_const = forward_mask.shape == (1,)  # The mask is redundant in this case
# mask_rows = jnp.sum(forward_mask, axis=-1, dtype=float)  # For non-trivial mask, precompute the scaling count
# eps = jnp.finfo(dft.dtype).eps # Large performance gain by avoiding the 0 branch of complex jnp.log()

# def evaluate(x: Array, weight: Array) -> Array:
#     """
#     Uses the convolution theorem to compute many ESP evaluations in parallel via the fourier domain.
#     O(n^2) before parallelisation - tree based convolution is O(nlog^2n), but requires ordering
#     and can't be cleanly parallelised.
#     """

#     # fmt: off
#     @jax.checkpoint
#     def fourier_checkpoint_stable(assignment: Array) -> Array:
#         # Add dimension to capture K+1 shifted roots for K terms of the clause.
#         fourier_domain = dft + assignment[:, None, :]  # (N?,(K+1),1) + (N,_,K) = (N,(K+1),K)

#         # For each frequency group (e.g. all terms shifted by some w^i), we scale back towards 1.
#         scaling = jnp.max(jnp.abs(fourier_domain), axis=-1, keepdims=True)
#         scaling = jnp.where(scaling == 0, 1.0, scaling)
#         fourier_scaled = fourier_domain / scaling

#         # Calculate rescaling terms required after summation/prod
#         scale_freq = jnp.squeeze(scaling, axis=-1)
#         times_freq = jnp.where(mask_const, fourier_scaled.shape[-1], mask_rows)

#         # Scaled Log domain calculations
#         fourier_log = jnp.log(fourier_scaled + eps)
#         fourier_logsum = jnp.sum(fourier_log, axis=-1, where=forward_mask)  # (N,(K+1))
#         fourier_rescale = fourier_logsum + (times_freq * jnp.log(scale_freq))
#         esp_freq = jnp.exp(fourier_rescale)

#         # # Scaled Time domain calculation (mutually exclusive with above block)
#         # fourier_prod = jnp.prod(fourier_scaled, axis=-1, where=forward_mask)  # (N,(K+1))
#         # esp_freq = fourier_prod * (scale_freq**times_freq)

#         esp_eval = idft * esp_freq
#         return esp_freq  # (N)

#     assignment = sign * x[lits]                                           # (N,K) * (N,K) = (N,K)
#     esp_freq = fourier_checkpoint_stable(assignment)                      # (N,(K+1))
#     esp_eval = idft * esp_freq                                            # (1,(K+1)) * (N,(K+1)) = (N,(K+1))
#     clause_eval = jnp.sum(esp_eval.real, axis=-1)                         # (N,)
#     weighted_eval = weight * clause_eval                                  # (N,) * (N,) = (N,)
#     x_eval = jnp.sum(weighted_eval, axis=-1)                              # (1,)
#     #print("Executing from python")
#     return jnp.atleast_1d(x_eval)
#     # fmt: on