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