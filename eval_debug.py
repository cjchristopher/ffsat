# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
# def eval_verify(
#     objs: tuple[Objective, ...],
# ) -> tuple[Callable[[Array, tuple[Array]], tuple[Array, Array]], Callable[[Array], Array]]:
#     def eval_verify_obj(obj: Objective) -> tuple[Callable[[Array, Array], Array], Callable[[Array], Array]]:
#         lits = obj.clauses.lits
#         sign = obj.clauses.sign
#         mask = obj.clauses.mask
#         cards = obj.clauses.cards
#         types = obj.clauses.types

#         dft, idft = obj.ffts
#         forward_mask = obj.forward_mask

#         def evaluate(xs: Array, weight: Array) -> Array:
#             """Let B = x points in batch, N = num clauses, K = length of clauses.
#             Uses the convolution theorem to compute many ESP evaluations in parallel via the fourier domain.
#             O(n^2) before parallelisation - tree based convolution is O(nlog^2n), but requires ordering
#             and can't be cleanly parallelised.
#             """

#             @jax.checkpoint
#             def fourier_checkpoint(assignment: Array) -> Array:
#                 # Add dimension to capture K+1 shifted roots for K terms of the clause.
#                 fourier_domain = dft + assignment[:, :, None, :]  # (N?,(K+1),1) + (B,N,_,K) = (B,N,(K+1),K)
#                 # jax.debug.print("JJ fourier {}\n{}", dft.shape[-1]+1, fourier_domain[0, 0, :5, :5], ordered=True)

#                 # For each frequency group (e.g. all terms shifted by some w^i), we scale back towards 1.
#                 scaling = jnp.max(jnp.abs(fourier_domain), axis=-1, keepdims=True)
#                 rect_scaling = jnp.where(scaling == 0, 1.0, scaling)
#                 fourier_scaled = fourier_domain / rect_scaling
#                 # jax.debug.print("JJ scaled \n{}", fourier_scaled[0, 0, :5, :5], ordered=True)

#                 # Calculate terms required to rescale after summation/prod
#                 scale_freq = jnp.squeeze(rect_scaling, axis=-1)
#                 times_freq = jnp.where(
#                     forward_mask.shape == (1,), fourier_scaled.shape[-1], jnp.sum(forward_mask, axis=-1)
#                 )
#                 # jax.debug.print("sizes {} {} {}", scale_freq.shape, times_freq.shape, (scale_freq**times_freq).shape, ordered=True)

#                 # Log domain calculations
#                 fourier_log = jnp.log(fourier_scaled)
#                 # jax.debug.print("JJ logged \n{}", fourier_log[0, 0, :5, :5], ordered=True)
#                 fourier_logsum = jnp.sum(fourier_log, axis=-1, where=forward_mask)  # (B,N,(K+1))
#                 # jax.debug.print("JJ sum \n{}", fourier_logsum[0, 0, :5], ordered=True)
#                 fourier_rescale = fourier_logsum + (times_freq * jnp.log(scale_freq))
#                 # jax.debug.print("JJ rescale \n{}\n{}\n{}", (times_freq, scale_freq[0,0,:5], jnp.log(scale_freq)[0,0,:5]), (times_freq * jnp.log(scale_freq))[0,0,:5], fourier_rescale[0, 0, :5], ordered=True)
#                 esp_freq = jnp.exp(fourier_rescale)
#                 # jax.debug.print("JJ esp_freq \n{}, {}", esp_freq[0, 0, :5], esp_freq[0, 0, -5:], ordered=True)

#                 # Time domain calculation
#                 # fourier_prod = jnp.prod(fourier_scaled, axis=-1, where=forward_mask) # (B,N,(K+1))
#                 # esp_freq = fourier_prod * (scale_freq**times_freq)
#                 # # jax.debug.print("JJ rescale \n{}\n{}\n{}", (times_freq, scale_freq[0,0,:5]), (scale_freq ** times_freq)[0,0,:5], fourier_rescale[0, 0, :5], ordered=True)
#                 # # jax.debug.print("JJ esp_freq \n{}, {}", esp_freq[0, 0, :5], esp_freq[0, 0, -5:], ordered=True)

#                 # lmax = assignment.shape[-1]
#                 esp_eval = idft * esp_freq
#                 # esp_eval = jnp.matmul(idft[None, None, ...], esp_freq[...,None])[...,0]
#                 # esp_eval = coeffs.reshape(-1) * jnp.einsum("bij,...j->...i", idft, esp_freq)#/(dft.shape[-1]+1)  # (N?,K+1) * (B,N,(K+1)) = (B,N,(K+1))
#                 # esp_eval = coeffs.reshape(-1)*jnp.matmul(idft.squeeze(), esp_freq[0,0,:].reshape(-1,1), precision=jax.lax.Precision.HIGHEST).reshape(1,1,-1)
#                 # correct_all_1 = jnp.array([comb(lmax, i) for i in range(lmax,-1, -1)])
#                 # jax.debug.print("JJ esp_eval \n{}", jnp.take_along_axis(esp_eval[0,0,:],
#                 #                                         jnp.argsort(
#                 #                                             jnp.abs(esp_eval[0,0,:]),axis=-1, descending=True),
#                 #                                         axis=-1), ordered=True)
#                 # jax.debug.print("JJ esp_eval2 \n{}\n{}\n{}", esp_eval[0, 0, :5],
#                 #                 esp_eval[0, 0, -5:],
#                 #                 jnp.sum(
#                 #                     jnp.take_along_axis(esp_eval[0,0,:],
#                 #                                         jnp.argsort(
#                 #                                             jnp.abs(esp_eval[0,0,:]),axis=-1, descending=True),
#                 #                                         axis=-1),
#                 #                         axis=-1), ordered=True)
#                 # jax.debug.print("JJ esp_eval3 \n{}\n{:.40e},{:.40e},{:.40e}", esp_eval[0, 0, 1:61] + esp_eval[0, 0, 61:][::-1], jnp.sum(esp_eval[0, 0, 1:61] + esp_eval[0, 0, 61:][::-1]), esp_eval[0,0,0], jnp.sum(esp_eval[0, 0, 1:61] + esp_eval[0, 0, 61:][::-1])+ esp_eval[0,0,0], ordered=True)
#                 return jnp.sum(esp_eval.real, axis=-1)  # (B,N)

#             assignment = sign * xs[:, lits]  # (N,K) * (B,N,K) = (B,N,K) (sign auto broadcasted)
#             clause_eval = fourier_checkpoint(assignment)  # (B,N,(K+1))
#             # clause_eval = jnp.sum(esp_eval, axis=-1)  # (B,N)
#             weighted_eval = weight * clause_eval  # (B,N) Clause weight if necessary
#             x_eval = jnp.sum(weighted_eval, axis=-1)  # (B,)

#             # jax.debug.print("\nJJ======================================S", ordered=True)
#             # jax.debug.print("JJ {}, {}, {}", types[0], reverse_dict, cards[0], ordered=True)
#             # jax.debug.callback(nan_check_2, "clause eval:\n", clause_eval, ordered=True)
#             # jax.debug.callback(nan_check_2, "esp eval:\n", esp_eval, ordered=True)
#             # jax.debug.callback(nan_check_2, "esp_freq_prod:\n", esp_freq_prod, ordered=True)
#             # jax.debug.callback(nan_check_2, "fourier:\n", fourier_domain, ordered=True)
#             # jax.debug.callback(nan_check_2, "assn:\n", assignment, ordered=True)
#             # jax.debug.print("JJ x {}\n{}", xs[0,:5], assignment[0, :3, :], ordered=True)
#             # jax.debug.print("JJ total cost per x {} {}", weighted_eval[0], x_eval[0], ordered=True)
#             # jax.debug.print("JJ======================================E\n", ordered=True)
#             return jnp.atleast_1d(x_eval)  # Ensure array for batch size of 1

#         def verify(xs: Array) -> Array:
#             assign = sign * xs[:, lits]  # (B,N,K)
#             unsat = jnp.zeros(assign.shape[:-1], dtype=bool)  # (B,N)

#             clause_unsat_rules = {
#                 "xor": lambda: jnp.sum(assign < 0, axis=-1, where=mask) % 2 == 0,
#                 "cnf": lambda: jnp.min(assign, axis=-1, where=mask, initial=jnp.inf) > 0,
#                 "eo": lambda: jnp.sum(assign < 0, axis=-1, where=mask) != 1,
#                 "amo": lambda: jnp.sum(assign < 0, axis=-1, where=mask) > 1,
#                 "nae": lambda: jnp.logical_not(
#                     jnp.logical_and(
#                         (jnp.min(assign, axis=-1, where=mask, initial=jnp.inf) < 0),
#                         (jnp.max(assign, axis=-1, where=mask, initial=-jnp.inf) > 0),
#                     )
#                 ),
#                 "card": lambda: jnp.where(
#                     cards < 0,
#                     jnp.sum(assign < 0, axis=-1, where=mask) >= jnp.abs(cards),
#                     jnp.sum(assign < 0, axis=-1, where=mask) < cards,
#                 ),
#             }

#             for clause_type, handler in clause_unsat_rules.items():
#                 type_id = clause_type_ids[clause_type]
#                 unsat_cond = handler()
#                 type_unsat = jnp.where(types == type_id, unsat_cond, unsat)
#                 # jax.debug.print("Assign: {} \ntype: {}, \nunsat: \n{}\n{}", (cards, assign[0,0,:]), clause_type, type_unsat, jnp.sum(assign < 0, axis=-1, where=mask), ordered=True)
#                 unsat = unsat | type_unsat

#             return jnp.atleast_2d(unsat)

#         return evaluate, verify

#     evaluators, verifiers = zip(*[eval_verify_obj(obj) for obj in objs])

#     def evaluate_assn(xs: Array, weights: tuple[Array]) -> tuple[Array, Array]:
#         # Each has shape (B,1), the cost for the Bth x against each clause set.
#         x_cost_per_obj = [evaluate(xs, weight) for (evaluate, weight) in zip(evaluators, weights)]

#         # Use jax.vmap to parallelize over evaluators (objectives)
#         # def eval_one(xs, evaluator, weight):
#         #     return evaluator(xs, weight)
#         # x_cost_per_obj = jax.vmap(eval_one, in_axes=(None, 0, 0))(xs, jnp.array(evaluators), jnp.array(weights))

#         # Stack these s.t. x_cost_stack[B] = [Bth x on cost_set_1, cost_set_2, etc]
#         x_cost_stack = jnp.stack(x_cost_per_obj, axis=-1)
#         x_costs = jnp.sum(x_cost_stack, axis=-1)
#         batch_cost = jnp.sum(x_costs)
#         return batch_cost  # , x_costs

#     def verify_assn(xs: Array) -> Array:
#         # e.g. [(B, N1), (B, N2), ...] for possibly different N1, N2, etc.
#         x_unsat_per_obj = [verify(xs) for verify in verifiers]
#         # Get (B, N1+N2+...)
#         x_unsat = jnp.concat(x_unsat_per_obj, axis=-1)
#         return x_unsat

#     return evaluate_assn, verify_assn

myglobal = 0

def nan_check(assignment, prod, clause_score, cost):
    if (
        jnp.any(jnp.isnan(assignment))
        or jnp.any(jnp.isnan(cost))
        or jnp.any(jnp.isnan(prod))
        or jnp.any(jnp.isnan(clause_score))
    ):
        x = f"assignment: {assignment}, prod: {prod}, clause score: {clause_score}, weighted {cost}"
        print(x)


def nan_check_2(array, name, clen):
    global myglobal
    clen = int(np.array(clen))
    host_array = np.array(array)
    print("nan check", clen, myglobal, name)
    if myglobal < 16 and clen > 5:
        jnp.save("w_" + name + str(myglobal//7), host_array, allow_pickle=False)
        myglobal+=7
    if jnp.any(jnp.isnan(host_array)):
        x = f"{name}: {host_array}"
        print(x)

def nan_check_2(string, array):
    global myglobal
    host_array = np.array(array)
    if (myglobal > 0 and myglobal < 5) or jnp.any(jnp.isnan(host_array)):
        myglobal = True
        x = f"{string} {host_array}\n"
        print(x)
