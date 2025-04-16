import jax
import jax.numpy as jnp
from jax import random, grad


@jax.jit
def langevin_step(x, T, rng, objective_fn, eta):
    rng, rng_noise = random.split(rng)
    noise = random.normal(rng_noise, shape=x.shape)
    grad_f = grad(objective_fn)(x)
    x_new = x - eta * grad_f + jnp.sqrt(2.0 * T) * noise
    x_new = jnp.clip(x_new, -1.0, 1.0)
    f_new = objective_fn(x_new)
    return x_new, f_new, rng


@jax.jit
def exchange_states(x, f, T, rng):
    def swap(i, carry):
        x, f, rng = carry
        rng, subrng = random.split(rng)
        delta = (f[i + 1] - f[i]) * (1 / T[i + 1] - 1 / T[i])
        swap_prob = jnp.exp(-delta)
        do_swap = random.uniform(subrng) < swap_prob

        xi, xip1 = jnp.where(do_swap, x[i + 1], x[i]), jnp.where(do_swap, x[i], x[i + 1])
        fi, fip1 = jnp.where(do_swap, f[i + 1], f[i]), jnp.where(do_swap, f[i], f[i + 1])
        x = x.at[i].set(xi).at[i + 1].set(xip1)
        f = f.at[i].set(fi).at[i + 1].set(fip1)
        return (x, f, rng), None

    (x, f, rng), _ = jax.lax.scan(swap, (x, f, rng), jnp.arange(x.shape[0] - 1))
    return x, f, rng


def parallel_tempering_langevin(
    objective_fn,
    x_init,  # [num_replicas, n]
    rng_key,
    T_init=1.0,
    T_final=1e-3,
    max_steps=1000,
    alpha=0.995,
    eta=0.01,
    swap_interval=10,
    enable_tempering=True,
):
    num_replicas, n = x_init.shape
    T = jnp.geomspace(T_init, T_final, num_replicas)  # fixed temps
    rng_keys = random.split(rng_key, num_replicas + 1)
    rng_global, rngs = rng_keys[0], rng_keys[1:]

    def body_fn(state, step):
        x, f, T, rng_global, rngs = state

        @jax.vmap
        def update(xi, Ti, ri):
            return langevin_step(xi, Ti, ri, objective_fn, eta)

        x, f, rngs = update(x, T, rngs)

        if enable_tempering:

            def swap_if_needed():
                return exchange_states(x, f, T, rng_global)

            x, f, rng_global = jax.lax.cond(step % swap_interval == 0, swap_if_needed, lambda: (x, f, rng_global))

        return (x, f, T, rng_global, rngs), f

    f_init = jax.vmap(objective_fn)(x_init)
    initial_state = (x_init, f_init, T, rng_global, rngs)
    (x_final, f_final, _, _, _), f_trace = jax.lax.scan(body_fn, initial_state, jnp.arange(max_steps))
    return x_final, f_final, f_trace
