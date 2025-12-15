# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import abc
import functools
import logging
from collections.abc import Callable
from time import perf_counter as time
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
# import optimistix as optx
from jax import Array
from jaxopt import LBFGSB, NonlinearCG, ProjectedGradient, ProximalGradient, ScipyBoundedMinimize
from jaxopt._src import base as job
from jaxopt.projection import projection_box as box
from numpy.typing import NDArray
# from optimistix import BoundedLBFGS
from scipy.optimize import Bounds, OptimizeResult
from scipy.optimize import minimize as ScipyMinimize

from boolean_whf import Objective, clause_type_ids

logger = logging.getLogger(__name__)


# TODO: Import specific solver modules when needed
# from . import hj_mad
# from . import langevin_annealing
# from . import pgd
print = functools.partial(print, flush=True)

EvalFn: TypeAlias = Callable[[Array, Array, Array], Array]
SeqEvalFn: TypeAlias = Callable[[Array, Array, tuple[Array]], Array | tuple[Array, Array | tuple[Array, Array]]]
VerifyFn: TypeAlias = Callable[[Array], Array]


def build_eval_verify(objs: tuple[Objective, ...]) -> tuple[tuple[EvalFn, ...], tuple[VerifyFn, ...]]:
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

    def single_eval_verify(obj: Objective) -> tuple[EvalFn, VerifyFn]:
        lits = obj.clauses.lits
        sign = obj.clauses.sign
        mask = obj.clauses.mask
        cards = obj.clauses.cards
        types = obj.clauses.types

        dft, idft = obj.ffts
        forward_mask = True if jnp.all(obj.forward_mask) else obj.forward_mask
        clause_count = lits.shape[0]
        # fmt: off
        unsat_rules = {
            "xor": lambda x: jnp.sum(x < 0, axis=1, where=mask) % 2 == 0,
            "cnf": lambda x: jnp.min(x, axis=1, where=mask, initial=jnp.inf) > 0,
            "eo":  lambda x: jnp.sum(x < 0, axis=1, where=mask) != 1,
            "amo": lambda x: jnp.sum(x < 0, axis=1, where=mask) > 1,
            "nae": lambda x: jnp.logical_not(
                                jnp.logical_and(
                                    (jnp.min(x, axis=1, where=mask, initial=jnp.inf) < 0),
                                    (jnp.max(x, axis=1, where=mask, initial=-jnp.inf) > 0),
                                )
                            ),
            "card": lambda x: jnp.where(
                                cards < 0,
                                jnp.sum(x < 0, axis=1, where=mask) >= jnp.abs(cards),
                                jnp.sum(x < 0, axis=1, where=mask) < cards,
                            ),
            "ek": lambda x: jnp.sum(x < 0, axis=1, where=mask) != cards,
        }

        def evaluate(x: Array, fixed_vars: Array, weight: Array) -> Array:
            #TODO: Create an eval_rules dict (like above unsat_rules) that only applies the full transform
            # for clause types that need it. e.g. XOR can be shortcut substantially if we close over the types.

            #TODO: Re-inline this function at some point. I'm no longer convinced jax.checkpoint helps us here.
            #@jax.checkpoint
            def fourier_checkpoint(x: Array) -> Array:
                assignment = sign * x[lits]                                       # (N,K) * (N,K) = (N,K)
                # Add dimension to capture K+1 shifted roots for K terms of the clause.
                fourier_domain = dft + assignment[:, None, :]                     # (N,(K+1),1) + (N,_,K) = (N,(K+1),K)
                esp_freq = jnp.prod(fourier_domain, axis=-1, where=forward_mask)
                return esp_freq                                                   # (N,(K+1))

            x = jnp.where(fixed_vars, jax.lax.stop_gradient(x), x)
            esp_freq = fourier_checkpoint(x)                                      # (N,(K+1))
            esp_eval = idft * esp_freq                                            # (1,(K+1)) * (N,(K+1)) = (N,(K+1))
            clause_eval = jnp.sum(esp_eval.real, axis=-1)                         # (N,)
            weighted_eval = weight * clause_eval                                  # (N,) * (N,) = (N,)
            x_eval = jnp.sum(weighted_eval, axis=-1)                              # (1,)
            return jnp.atleast_1d(x_eval)
        # fmt: on

        def verify(x: Array) -> Array:
            assignment = sign * x[lits]
            unsat = jnp.zeros(clause_count, dtype=bool)

            for clause_type, rule in unsat_rules.items():
                type_id = clause_type_ids[clause_type]
                unsat_clauses = rule(assignment)
                unsat = unsat | jnp.where(types == type_id, unsat_clauses, unsat)

            return unsat

        return evaluate, verify

    eval_fns: tuple[EvalFn]
    verify_fns: tuple[VerifyFn]
    eval_fns, verify_fns = zip(*[single_eval_verify(obj) for obj in objs])
    return eval_fns, verify_fns


def seq_eval_verify(eval_fns: tuple[EvalFn, ...], verify_fns: tuple[VerifyFn, ...]) -> tuple[SeqEvalFn, VerifyFn]:
    """
    Groups a collection (usually all) of Evaluation functions & Verifier functions into a sequence.
    """

    def seq_evals(x: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> tuple[Array, Array | tuple[Array, Array]]:
        costs = [evaluate(x, fixed_vars, weight) for (evaluate, weight) in zip(eval_fns, weights)]
        cost = jnp.sum(jnp.array(costs))
        return cost, (x, cost)  # returns costs in aux for breakdown by objective. aux info - remove when consolidating

    def seq_verifies(x: Array) -> Array:
        all_res = [verify(x) for verify in verify_fns]
        res = jnp.concat(all_res, axis=-1)
        return res

    return seq_evals, seq_verifies


class FFSatSolver(abc.ABC):
    def __init__(
        self,
        evaluator: SeqEvalFn,
        verifier: VerifyFn,
        solver: str = "lbfgsb",
        maxiter: int = 100,
    ) -> None:
        self.sol_name = solver
        self.maxiter = maxiter
        self.warmup_sol = False
        self.memest = None

        # TODO: Change to optimistix when mature.
        # if self.sol_name in ["optim"]:
        #     self.solver = BoundedBFGS()
        #     self.solver = optimistix.minimise(evaluator, BoundedBFGS, args={'weights': weights}, has_aux=True)

        if self.sol_name in ["lbfgsb", "pgd", "josp-lbfgsb"]:  # probably change to if sol_name in JAX_OPTIMS
            if self.sol_name in ["lbfgsb"]:
                bfgsb = LBFGSB(fun=evaluator, maxiter=self.maxiter, has_aux=True)
                logger.info("Setting up JAXOPT L-BFGS-B:", type(self.solver))

                def opt(x: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> tuple[Array, Array, Array, Array]:
                    bounds = (-1 * jnp.ones_like(x), jnp.ones_like(x))
                    x_opt, state = bfgsb.run(init_params=x, fixed_vars=fixed_vars, weights=weights, bounds=bounds)
                    unsat = jnp.squeeze(verifier(x_opt))
                    return x_opt, unsat, jnp.atleast_1d(state.iter_num), state.aux

            elif self.sol_name in ["josp-lbfgsb"]:
                spminB = ScipyBoundedMinimize(fun=evaluator, method="L-BFGS-B", maxiter=self.maxiter, has_aux=True)
                logger.info("Setting up JAXOPT ScipyBounded L-BFGS-B")

                def opt(x: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> tuple[Array, Array, Array, Array]:
                    bounds = (-1 * jnp.ones_like(x), jnp.ones_like(x))
                    x_opt, state = spminB.run(x, fixed_vars=fixed_vars, weights=weights, bounds=bounds)
                    unsat = jnp.squeeze(verifier(x_opt))
                    return x_opt, unsat, jnp.atleast_1d(state.iter_num), state.fun_val

            elif self.sol_name in ["pgd"]:
                pgd = ProjectedGradient(fun=evaluator, projection=box, maxiter=self.maxiter, has_aux=True)
                logger.info("Setting up JAXOPT Projected Gradient (Box)")

                def opt(x: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> tuple[Array, Array, Array, Array]:
                    x_opt, state = pgd.run(x, fixed_vars=fixed_vars, weights=weights, hyperparams_proj=(-1, 1))
                    unsat = jnp.squeeze(verifier(x_opt))
                    return x_opt, unsat, jnp.atleast_1d(state.iter_num), state.aux

            else:
                pass

            def vectorise(
                xs: Array, fixed_vars: Array, weights: tuple[Array, ...]
            ) -> tuple[Array, Array, Array, Array, Array]:
                x_opt, unsat, iters, evals = jax.vmap(opt, in_axes=(0, 0, None))(xs, fixed_vars, weights)
                # jax.debug.print("min/max evals {}, {}, {}", jnp.min(evals), jnp.max(evals), evals.shape)
                unsat_cl_count = jnp.sum(jnp.atleast_1d(unsat), axis=1)
                return x_opt, jnp.atleast_2d(unsat), iters, unsat_cl_count, evals

            self.run = jax.jit(vectorise, donate_argnums=(0))

        elif self.sol_name in ["prox"]:
            self.solver = ProximalGradient
            # self.solver = ProximalGradient(fun=evaluator, prox=hj_moreau, maxiter=self.maxiter)
            raise NotImplementedError("HJ Moreau Proximal Gradient not yet Implemented")

        elif self.sol_name in ["nlcg"]:
            self.solver = NonlinearCG
            raise NotImplementedError("Non-Linear Conjugate Gradient not yet Implemented")

        elif self.sol_name in ["langevin"]:
            raise NotImplementedError("Langevin Annealing not yet Implemented")

        elif self.sol_name in ["sp-lbfgsb"]:
            # We can only sensibly run this in combined multi-start since sp.minimize is max(CPU-thread) bound.
            # Here we manually VMAP and JIT the opt and ver steps
            self.solver = ScipyMinimize
            logger.info("Setting up ScipyBounded L-BFGS-B (CPU) with JIT'd eval+verify (GPU)")
            _, _, eval_fun = job._make_funs_without_aux(fun=evaluator, value_and_grad=False, has_aux=True)
            v_eval_fun = jax.jit(jax.vmap(eval_fun, in_axes=(0, 0, None)))
            v_verifier = jax.jit(jax.vmap(verifier, in_axes=(0)))
            self.eval_fun = v_eval_fun

            def full_vectorise(
                x: Array, fixed_vars: Array, weights: tuple[Array, ...]
            ) -> tuple[Array, Array, Array, Array, Array]:
                def flat(np_x0: NDArray) -> tuple[float, NDArray]:
                    # Convert back to JAX arrays, run on GPU, return CPU results to minimize.
                    x0 = jnp.array(np_x0).reshape(x.shape)
                    v, g = v_eval_fun(x0, fixed_vars, weights)  # this is eval.
                    return float(jnp.sum(v)), np.array(g.flatten())

                x0 = np.array(x.flatten())
                bounds = Bounds(-1, 1, True)
                options = {"maxiter": self.maxiter}
                res: OptimizeResult = ScipyMinimize(flat, x0, bounds=bounds, jac=True, options=options)
                x_opt = jnp.array(res.x).reshape(x.shape)
                unsat = jnp.squeeze(v_verifier(x_opt))
                unsat_cl_count = jnp.sum(jnp.atleast_1d(unsat), axis=0)
                return x_opt, unsat, jnp.array([res.nit]), unsat_cl_count, jnp.array(res.aux)

            self.run = full_vectorise

        else:
            pass

    def peak_memory_estimation(self, x0: Array, fixed_vars: Array, weights: tuple[Array, ...]) -> int:
        runner = self.run if self.sol_name not in ["sp-lbfgsb"] else self.eval_fun
        traced = runner.trace(x0, fixed_vars, weights)
        lowered = traced.lower()
        compiled = lowered.compile()
        analysis = compiled.memory_analysis()

        peak_est = (
            analysis.temp_size_in_bytes  # type: ignore
            + analysis.argument_size_in_bytes  # type: ignore
            + analysis.output_size_in_bytes  # type: ignore
            - analysis.alias_size_in_bytes  # type: ignore
        )
        return int(peak_est)

    def warmup(self, warmup_data: tuple[Array, Array, tuple[Array, ...]], counting: bool = False) -> None:
        if warmup_data:
            runner = self.run if self.sol_name not in ["sp-lbfgsb"] else self.eval_fun
            x0, fixed_vars, weights = warmup_data
            logger.info("Warmup Run (Dummy Data Compilation)")
            t0 = time()
            opt_x0, opt_unsat, _, _, _ = runner(x0, fixed_vars, weights)
            if not counting:
                batch_unsat_scores = jnp.sum(opt_unsat, axis=1)
                batch_best = jnp.min(batch_unsat_scores)
                loc = jnp.argmin(batch_unsat_scores)
                best_x = opt_x0[loc]
                if batch_best == 0:
                    print("Found a solution in warmup! SAT at index {}".format(loc))
                    out_string = "v"
                    assignment = []
                    for i in range(x0.shape[-1]):
                        lit = i + 1
                        if best_x[i] > 0:
                            out_string += " {}".format(-lit)
                            assignment.append(-lit)
                        else:
                            out_string += " {}".format(lit)
                            assignment.append(lit)
                    print(out_string)
                    self.warmup_sol = True

            logger.info("Warmup Complete", time() - t0)
