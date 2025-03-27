# ------------------------------------------------------------------------------------------------------------
# HJ Moreau Adaptive Descent
# ------------------------------------------------------------------------------------------------------------
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike as Array


class HJ_MAD:
    """
    Hamilton-Jacobi Moreau Adaptive Descent (HJ_MAD) is used to solve nonconvex minimization
    problems via a zeroth-order sampling scheme.

    Inputs:
      1)  f            = function to be minimized. Inputs have size (n_samples x n_features). Outputs have size n_samples
      2)  x_true       = true global minimizer
      3)  delta        = coefficient of viscous term in the HJ equation
      4)  int_samples  = number of samples used to approximate expectation in heat equation solution
      5)  x_true       = true global minimizer
      6)  t_vec        = time vector containig [initial time, minimum time allowed, maximum time]
      7)  max_iters    = max number of iterations
      8)  tol          = stopping tolerance
      9)  theta        = parameter used to update tk
      10) beta         = exponential averaging term for gradient beta (beta multiplies history, 1-beta multiplies current grad)
      11) eta_vec      = vector containing [eta_minus, eta_plus], where eta_minus < 1 and eta_plus > 1 (part of time update)
      11) alpha        = step size. has to be in between (1-sqrt(eta_minus), 1+sqrt(eta_plus))
      12) fixed_time   = boolean for using adaptive time
      13) verbose      = boolean for printing

    Outputs:
      1) x_opt                    = optimal x_value approximation
      2) xk_hist                  = update history
      3) tk_hist                  = time history
      4) fk_hist                  = function value history
      5) xk_error_hist            = error to true solution history
      6) rel_grad_uk_norm_hist    = relative grad norm history of Moreau envelope
    """

    def __init__(
        self,
        f,
        x_true,
        delta=0.1,
        int_samples=100,
        t_vec=[1.0, 1e-3, 1e1],
        max_iters=5e4,
        tol=5e-2,
        theta=0.9,
        beta=[0.9],
        eta_vec=[0.9, 1.1],
        alpha=1.0,
        fixed_time=False,
        verbose=True,
    ):
        self.delta = delta
        self.f = f
        self.int_samples = int_samples
        self.max_iters = max_iters
        self.tol = tol
        self.t_vec = t_vec
        self.theta = theta
        self.x_true = x_true
        self.beta = beta
        self.alpha = alpha
        self.eta_vec = eta_vec
        self.fixed_time = fixed_time
        self.verbose = verbose

        # check that alpha is in right interval
        assert alpha >= 1 - jnp.sqrt(eta_vec[0])
        assert alpha <= 1 + jnp.sqrt(eta_vec[1])

    def compute_grad_uk(self, x: Array, t, f, delta, eps=1e-12):
        """
        compute gradient of Moreau envelope
        """

        standard_dev = jnp.sqrt(delta * t)

        n_features = x.shape[0]
        key = jax.random.key(0)
        xRand = jax.random.truncated_normal(key, -1, 1, (self.int_samples, n_features))
        y = standard_dev * xRand + x

        exp_term = jnp.exp(-f(y) / delta)
        v_delta = jnp.mean(exp_term)

        # separate grad_v into two terms for numerical stability
        numerator = y * exp_term.reshape(self.int_samples, 1)
        numerator = jnp.mean(numerator, axis=0)
        grad_uk = x - numerator / (v_delta + eps)  # the t gets canceled with the update formula

        uk = -delta * jnp.log(v_delta + eps)

        return grad_uk, uk

    def update_time(self, tk, rel_grad_uk_norm):
        """
        time step rule

        if ‖gk_plus‖≤ theta (‖gk‖+ eps):
          min (eta_plus t,T)
        else
          max (eta_minus t,t_min) otherwise

        OR:

        if rel grad norm too small, increase tk (with maximum T).
        else if rel grad norm is too "big", decrease tk with minimum (t_min)
        """

        eta_minus = self.eta_vec[0]
        eta_plus = self.eta_vec[1]
        T = self.t_vec[2]
        t_min = self.t_vec[1]

        if rel_grad_uk_norm <= self.theta:
            # increase t when relative gradient norm is smaller than theta
            tk = min(eta_plus * tk, T)
        else:
            # decrease otherwise t when relative gradient norm is smaller than theta
            tk = max(eta_minus * tk, t_min)

        return tk

    def run(self, x0: Array):
        n_features = x0.shape[0]

        xk_hist = jnp.zeros(self.max_iters, n_features)
        xk_error_hist = jnp.zeros(self.max_iters)
        rel_grad_uk_norm_hist = jnp.zeros(self.max_iters)
        fk_hist = jnp.zeros(self.max_iters)
        tk_hist = jnp.zeros(self.max_iters)
        counter = 1

        xk = x0
        x_opt = xk
        tk = self.t_vec[0]
        t_max = self.t_vec[2]

        first_moment, _ = self.compute_grad_uk(xk, tk, self.f, self.delta)
        rel_grad_uk_norm = 1.0

        fmt = "[{:3d}]: fk = {:6.2e} | xk_err = {:6.2e} "
        fmt += " | |grad_uk| = {:6.2e} | tk = {:6.2e}"

        print("-------------------------- RUNNING HJ-MAD ---------------------------")
        print("dimension = ", n_features, "n_samples = ", self.int_samples)

        for k in range(self.max_iters):
            xk_hist[k, :] = xk

            rel_grad_uk_norm_hist[k] = rel_grad_uk_norm

            xk_error_hist[k] = jnp.linalg.norm(xk - self.x_true)
            tk_hist[k] = tk

            fk_hist[k] = self.f(xk.reshape(1, n_features))

            if self.verbose:
                print(fmt.format(k + 1, fk_hist[k], xk_error_hist[k], rel_grad_uk_norm_hist[k], tk))

            if xk_error_hist[k] < self.tol:
                tk_hist = tk_hist[0 : k + 1]
                xk_hist = xk_hist[0 : k + 1, :]
                xk_error_hist = xk_error_hist[0 : k + 1]
                rel_grad_uk_norm_hist = rel_grad_uk_norm_hist[0 : k + 1]
                fk_hist = fk_hist[0 : k + 1]
                print("HJ-MAD converged with rel grad norm {:6.2e}".format(rel_grad_uk_norm_hist[k]))
                print("iter = ", k, ", number of function evaluations = ", len(xk_error_hist) * self.int_samples)
                break
            elif k == self.max_iters - 1:
                print("HJ-MAD failed to converge with rel grad norm {:6.2e}".format(rel_grad_uk_norm_hist[k]))
                print("iter = ", k, ", number of function evaluations = ", len(xk_error_hist) * self.int_samples)
                print("Used fixed time = ", self.fixed_time)

            if k > 0:
                if fk_hist[k] < fk_hist[k - 1]:
                    x_opt = xk

            xk = xk - self.alpha * first_moment  # tk gets canceled out with gradient formula

            if not self.fixed_time:
                tk = self.update_time(tk, rel_grad_uk_norm)

            grad_uk, _ = self.compute_grad_uk(xk, tk, self.f, self.delta)

            grad_uk_norm_old = jnp.linalg.norm(first_moment)
            first_moment = self.beta * first_moment + (1 - self.beta) * grad_uk

            grad_uk_norm = jnp.linalg.norm(first_moment)
            rel_grad_uk_norm = grad_uk_norm / (grad_uk_norm_old + 1e-12)

        return x_opt, xk_hist, tk_hist, xk_error_hist, rel_grad_uk_norm_hist, fk_hist


def compute_prox(x: Array, t, f, delta=1e-1, int_samples=100, alpha=1.0, linesearch_iters=0):
    """Estimate proximals from function value sampling via HJ-Prox Algorithm.

    The output estimates the proximal:

    $$
        \mathsf{prox_{tf}(x) = argmin_y \ f(y) + \dfrac{1}{2t} \| y - x \|^2,}
    $$

    where $\mathsf{x}$ = `x` is the input, $\mathsf{t}$=`t` is the time parameter,
    and $\mathsf{f}$=`f` is the function of interest. The process for this is
    as follows.

    - [x] Sample points $\mathsf{y^i}$ (via a Gaussian) about the input $\mathsf{x}$
    - [x] Evaluate function $\mathsf{f}$ at each point $\mathsf{y^i}$
    - [x] Estimate proximal by using softmax to combine the values for $\mathsf{f(y^i)}$ and $\mathsf{y^i}$

    Note:
        The computation for the proximal involves the exponential of a potentially
        large negative number, which can result in underflow in floating point
        arithmetic that renders a grossly inaccurate proximal calculation. To avoid
        this, the "large negative number" is reduced in size by using a smaller
        value of alpha, returning a result once the underflow is not considered
        significant (as defined by the tolerances "tol" and "tol_underflow").
        Utilizing a scaling trick with proximals, this is mitigated by using
        recursive function calls.

    Warning:
        Memory errors can occur if too many layers of recursion are used,
        which can happen with tiny delta and large f(x).

    Args:
        x (tensor): Input vector
        t (tensor): Time > 0
        f (Callable): Function to minimize
        delta (float, optional): Smoothing parameter
        int_samples (int, optional): Number of samples in Monte Carlo sampling for integral
        alpha (float, optional): Scaling parameter for sampling variance
        linesearch_iters (int, optional): Number of steps used in recursion (used for numerical stability)
        device (string, optional): Device on which to store variables

    Shape:
        - Input `x` is of size `(n, 1)` where `n` is the dimension of the space of interest
        - The output `prox_term` also has size `(n, 1)`

    Returns:
        prox_term (tensor): Estimate of the proximal of f at x
        linesearch_iters (int): Number of steps used in recursion (used for numerical stability)
        envelope (tensor): Value of envelope function (i.e. infimal convolution) at proximal

    Example:
        Below is an exmaple for estimating the proximal of the L1 norm. Note the function
        must have inputs of size `(n_samples, n)`.
        ```
            def f(x):
                return torch.norm(x, dim=1, p=1)
            n = 3
            x = torch.randn(n, 1)
            t = 0.1
            prox_term, _, _ = compute_prox(x, t, f, delta=1e-1, int_samples=100)
        ```
    """
    assert x.shape[1] == 1
    assert x.shape[0] >= 1

    linesearch_iters += 1
    standard_dev = jnp.sqrt(delta * t / alpha)
    dim = x.shape[0]

    key = jax.random.key(0)
    xRand = jax.random.truncated_normal(key, -1, 1, (int_samples, dim))
    y = standard_dev * xRand + x.T  # y has shape (n_samples, dim)
    z = -f(y) * (alpha / delta)  # shape =  n_samples
    w = jax.nn.softmax(z, dim=0)  # shape = n_samples

    softmax_overflow = 1.0 - (w < jnp.inf).prod()
    if softmax_overflow:
        alpha *= 0.5
        return compute_prox(
            x, t, f, delta=delta, int_samples=int_samples, alpha=alpha, linesearch_iters=linesearch_iters, device=device
        )
    else:
        prox_term = jnp.matmul(w.t(), y)
        prox_term = prox_term.view(-1, 1)

    prox_overflow = 1.0 - (prox_term < jnp.inf).prod()
    assert not prox_overflow, "Prox Overflowed"

    envelope = f(prox_term.view(1, -1)) + (1 / (2 * t)) * jnp.linalg.norm(prox_term - x.T, ord=2) ** 2
    return prox_term, linesearch_iters, envelope
