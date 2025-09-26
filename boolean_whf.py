from __future__ import annotations

import operator as ops
from fractions import Fraction
from itertools import accumulate
from math import comb
from typing import NamedTuple
from typing import Optional as Opt

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from jax import Array
from scipy.stats import binom

from diskcache import FFSATCache

type Clause = list[int]
type Clauses = list[Clause]


class ClauseSignature(NamedTuple):
    type: str
    len: int
    card: int


class FFT(NamedTuple):
    dft: Array
    idft: Array


class ClauseArrays(NamedTuple):
    lits: Array  # = jnp.empty((0, 0), dtype=int)
    sign: Array  # = jnp.empty((0, 0), dtype=int)
    mask: Array  # = jnp.empty((0, 0), dtype=bool)
    types: Array  # = jnp.empty((0, 0), dtype=int)
    cards: Array  # = jnp.empty((0, 0), dtype=int)


class Objective(NamedTuple):
    clauses: ClauseArrays
    ffts: FFT
    forward_mask: Array


clause_type_ids: dict[str, int] = {"xor": 1, "eo": 2, "nae": 3, "cnf": 4, "amo": 5, "card": 0}


# class ClauseGroup:
class ClauseProcessor:
    def __init__(self, n_devices: int = 1, disk_cache: FFSATCache = None) -> None:
        self.n_devices: int = n_devices
        self.live_cache: dict = {}
        self.disk_cache: FFSATCache | None = None

        if disk_cache and isinstance(disk_cache, FFSATCache):
            self.disk_cache: dict = disk_cache

    def _wf_coeffs(self, sig: ClauseSignature) -> Array:
        """Calculate Walsh-Fourier coefficients for a given clause type.
        Returns the c_len+1 coefficients as per:
        [Constant, Degree 1 ESP, Degree 2 ESP, ..., Degree C_LEN ESP]
        """

        def __int_noisepoly_numerators(n: int, k: int, neg_k: int) -> list[int]:
            """Numpy's poly.poly.polyfromroots uses convolutions and changes to float to account for polynomial
            construction from generic roots. For even moderate n (e.g. 120), this causes numerical errors.
            Here our polynomial is monic, and only has roots in {-1, 1}, and so we can use the binomial theorem
            and symmetry explotiation to be faster, and more accurate.

            Compute the coefficients of a polynomial roots -1 ($$ n-k $$ mutiplicity) and 1 ($$ k-1 $$ multiplicity).
            Factor in the $$(-1)^{k-1}$$ and the combinatoric constant multiplier to get the final numerators.
            """
            if k < 1 or k > n:
                raise ValueError("k must satisfy 1 ≤ k ≤ n")

            coeffs = [0] * n
            for i in range((n - k) + 1):
                for j in range((k - 1) + 1):
                    power = i + j
                    if power > (n - 1) // 2:
                        break
                    coeff = comb((n - k), i) * comb((k - 1), j)
                    if (k - 1 - j) % 2 == 1:
                        coeff = -coeff
                    coeffs[power] += coeff
            sign = (-1) ** (k - 1)

            for i in range((n - 1) // 2 + 1):
                mirror_idx = (n - 1) - i
                if mirror_idx != i:
                    coeffs[mirror_idx] = sign * coeffs[i]

            negate = neg_k * ((-1) ** ((k - 1) % 2))
            const = comb(n - 1, k - 1)
            coeffs = [const * (negate * coeff) for coeff in coeffs]
            assert all([abs(c1) == abs(c2) for c1, c2 in zip(coeffs, coeffs[::-1])])
            return coeffs

        def __cnf(n: int) -> NDArray:
            r"""Walsh-Fourier coefficients for CNF (OR) clauses.
            $$
            \widehat{\texttt{OR}}(S) = \left\lbrace
            \begin{aligned}
                &\frac{1}{2^{n-1}} - 1 & \abs{S} = 0 \\
                &\frac{1}{2^{n-1}} & \abs{S} \neq 0
            \end{aligned}\right.
            $$
            """
            d = np.full(n + 1, 1 / (2 ** (n - 1)))
            d[0] -= 1
            return d

        def __eo(n: int) -> NDArray:
            r"""Walsh-Fourier coefficients for exactly-one (EO) clauses.
            $$
            \widehat{\texttt{EO}}(S) = \left\lbrace
            \begin{aligned}
                &1-\frac{n}{2^{n-1}} & \abs{S} = 0 \\
                &\frac{2\abs{S}-n}{2^{n-1}} & \abs{S} \neq 0
            \end{aligned}\right.
            $$
            """
            d = np.arange(-n, n + 1, 2, dtype=float) / (2 ** (n - 1))  # EO(S) = (2|S|-n)/2^(n-1)
            d[0] = 1 - n / (2 ** (n - 1))
            return d

        def __nae(n: int) -> NDArray:
            r"""Walsh-Fourier coefficients for not-all-equal (NAE) clauses.
            For all even $$|S|$$, the coefficient at that size is $$(1/2)^{n-2}$$ due to cancellation of all even terms in $$g(\rho)$$,
            and exactly $${n-1\choose |S|-1}$$ at all odd terms. This cancels the denominator in the general equation,
            leaving a factor of 2. This happens to essentially align with the $$|S|=0$$ case, but subtract 1. Hence:
            """
            d = np.zeros(n + 1, dtype=float)
            d[::2] = 1 / (2 ** (n - 2))
            d[0] -= 1
            return d

        def __xor(n: int) -> NDArray:
            r"""Walsh-Fourier coefficients for XOR clauses.
            All zero expect for the full product of all literals (The nth ESP in n variables).
            """
            d = np.zeros(n + 1, dtype=float)
            d[-1] = 1
            return d

        def __amo(n: int) -> NDArray:
            r"""Walsh-Fourier coefficients for at-most-one (AMO) clauses.
            $$
            \widehat{\texttt{AMO}}(S) = \left\lbrace
            \begin{aligned}
                &1-\frac{n-1}{2^{n-1}} & \abs{S} = 0 \\
                &\frac{2\abs{S}-n-1}{2^{n-1}} & \abs{S} \neq 0
            \end{aligned}\right.
            $$
            """

            d = np.arange(-n - 1, n, 2, dtype=float) / (2 ** (n - 1))
            d[0] = 1 - (n + 1) / (2 ** (n - 1))
            return d

        def __card(n: int, k: int) -> NDArray:
            r"""Walsh-Fourier coefficients for cardinality clauses.
            This does not simplify to a clean closed form, so we require some manual computation.
            The dominating step is polyfromroots, at $$O(n\cdot log^{2}(n))$$ for a clause of $$n$$ variables.
            $$
            \widehat{{\texttt{CARD}_{\geq k}}}(S) = \left\lbrace
            \begin{aligned}
                &1-\frac{\sum_{i=k}^{n}{n\choose i}}{2^{n-1}} & \abs{S} = 0 \\
                &\frac{{n-1 \choose k-1}\left(g_{\texttt{CARD}}(\rho)\right)_{[p^{|S|-1}]}}{{n-1\choose |S|-1}2^{n-1}} & \abs{S} \neq 0
            \end{aligned}\right.
            $$
            and $$ g_{\texttt{CARD}}(\rho) = (1+\rho)^{n-k}(1-\rho)^{k-1} $$

            For $$n=k$$, e.g. clause $$c = \texttt{CARD}^{\geq n}(x_1, x_2,\ldots, x_n)$$ is just $$c = x_1 \land x_2 \land \ldots \land x_n$$.
            Then $$g_{\texttt{CARD}} = (1-\rho)^{n-1}$$. The coefficients are just the  $$(n-1)$$th row of Pascal's triangle, 
            with every second term negated, or... $$(-1)^{|S|-1}{n-1 \choose |S|-1}$$, cancelling out.
            This means we get a simplified form:
            $$
            \widehat{{\texttt{CARD}_{*}}}(S) = \left\lbrace
            \begin{aligned}
                &1-\frac{1}{2^{n-1}} & \abs{S} = 0 \\
                &\frac{(-1)^{|S|-1}}{2^{n-1}} & \abs{S} \neq 0
            \end{aligned}\right.
            $$
            Having said this... there is merit to force reincoding this as n separate unit literals.
            Notably those individual terms will be trivially minimisable. This form is distinctly not.
            Also we need to still compute the full transformation in $$n$$ variables if we use this form.
            For $$k=0$$ the clause is trivial and should be dropped instead of evaluated as well - e.g
            clause $$c = \texttt{CARD}^{\geq 0}(x_1, x_2,\ldots)$$ is just $$FE_c = -1$$ (always SAT).
            These are now handled in the loader itself, but for posterity:
            $$n = k \Rightarrow$$ d = np.full(n + 1, (2 ** -(n - 1)), dtype=float); d[::2] *= -1; d[0] += 1
            $$k=0\Rightarrow$$ d = np.zeros(n + 1, dtype=float); d[0] = -1
            """
            neg_k = k / (abs(k))
            k = abs(k)
            # Calculate $$g(\rho) = (1+\rho)^{n-k}(1-\rho)^{k-1}$$, and reformulate for zeroes: $$(1-\rho) = -(\rho-1)$$
            # Therefore $$g(\rho) = (\rho+1)^{n-k}(\rho-1)^{k-1}(-1)^{k-1}$$, and then multiply by combinatoric term $${n-1 \choose k-1}$$
            # We do this ourselves from the $$\pm1$$ roots for numeric stability.
            coeff_numers = __int_noisepoly_numerators(n, k, neg_k)

            # The denominator for each $$|S|$$ has a $${n-1 \choose |S|-1}$$ term. Since each subsequent term in the series can be expressed
            # as the product of the last term and the current term $$S_i$$: $$S_{i+1} = S_i\cdot(n-i)/(i+1)$$, we save time computing the
            # m = (n-1)th row of Pascal's triangle (where the top is row 0) progressively (vectorised) vs calling comb()
            # The triangle is left-right symmetric, so m//2+1 is enough. Stay with native int as long as possible.
            m = n - 1
            offset = (m - 1) % 2

            # Calculate the factorial numerator and denominators separately, and integer divide for stability.
            # This implements the above recurrence for each subsequent term up to the midpoint of the triangle.
            # Then flip and concat - offset in case of unique middle element - and finally scale by $$2^{n-1}$$
            pascal_idx = list(range(1, ((m // 2) + 1)))
            pascal_denoms = list(accumulate(pascal_idx, ops.mul))  # cumprod (i*(i+1)*(i+2)...)
            pascal_numers = list(accumulate([n - p for p in pascal_idx], ops.mul))  # cumprod ((n-1)*(n-2)*...)
            coeff_denoms = [1] + [numer // denom for numer, denom in zip(pascal_numers, pascal_denoms)]
            coeff_denoms = [coeff * (2**m) for coeff in (coeff_denoms + coeff_denoms[::-1][offset:])]

            # Old numpy impl. Better to just stay in native infinite precision ints.
            # dtype = np.object_ if n > 64 else np.int64
            # pascal_numers = np.ones((m//2) + 1, dtype=dtype)
            # pascal_denoms = np.cumprod(pascal_numers[1:], dtype=dtype)
            # np.cumprod((m + 1 - pascal_numers[1:]), out=pascal_numers[1:], dtype=dtype)
            # np.floor_divide(pascal_numers[1:], pascal_denoms, out=pascal_numers[1:])
            # coeff_denoms = np.concatenate([coeff_denoms, coeff_denoms[::-1][offset:]], dtype=dtype) * 2**m
            # d = np.zeros(n + 1, dtype=float)
            # d[0] = 2 * binom.cdf(k - 1, n, 0.5) - 1
            # d[1:] = (coeff_numers / coeff_denoms)
            # assert all(np.abs(d[1:]) == np.abs(d[1:][::-1])) # Check symmetry

            # Compute the coefficients and check symmetry once more up to signs.
            coeffs = [numer / denom for numer, denom in zip(coeff_numers, coeff_denoms)]
            assert all([abs(c1) == abs(c2) for c1, c2 in zip(coeffs, coeffs[::-1])])

            d = np.array([2 * binom.cdf(k - 1, n, 0.5) - 1] + coeffs)
            return d

        method_map = {
            "cnf": __cnf,
            "eo": __eo,
            "nae": __nae,
            "xor": __xor,
            "amo": __amo,
            "card": __card,
        }

        if sig.type not in method_map:
            raise ValueError(f"Unknown clause type: {sig.type}")

        if sig.type == "card":
            if sig.card:
                return method_map[sig.type](sig.len, sig.card)
            # Theoretically unreachable
            raise ValueError("Cardinality clause without specified cardinality")
        return method_map[sig.type](sig.len)

    def _pad(self, arrays: list[NDArray | list[int]], pad_val: Opt[int] = 0) -> tuple[NDArray, NDArray]:
        arrays = [np.array(arr) if isinstance(arr, (list, int)) else arr for arr in arrays]
        rows = len(arrays)
        if rows > 1:
            #extra_rows = -rows % self.n_devices
            # TODO: Think about how to handle heterogeneous clause sets when objective sharding is required?
            # N.B. If we ever need to shard objectives then the above will be required for hetero batches.
            # For homogenous batches (1 signature), we would need to adjust the shard spec
            # This would indicate the mesh and shard spec should be the first thing we do and should
            # get passed through to this stage.
            # Or perhaps we just call _pad when we are doing the batch size detection. If we determine
            # that objective sharding is relevant, then we can pad then? Notably, we already
            # use n_devices here, so sequencing is something to think about.
            # This will affect types and cards as well. Those leaves will need replication for homogenous
            # objectives, instead of sharding. A fully heterogeneous batch will need sharding at all leaves.
            extra_rows = 0
        else:
            extra_rows = 0
        dtype = np.result_type(*arrays, pad_val)

        if arrays[0].ndim <= 1:
            max_length = max(len(arr) for arr in arrays)
            padded_shape = (rows + extra_rows, max_length)
            padded = np.full(padded_shape, pad_val, dtype=dtype)
            mask = np.zeros_like(padded, dtype=bool)

            for i, arr in enumerate(arrays):
                row_len = len(arr)
                padded[i, :row_len] = arr.ravel()
                mask[i, :row_len] = True

        else:
            try:
                assert np.all([arr.ndim == arrays[0].ndim for arr in arrays])
            except AssertionError as e:
                raise e("All sub arrays must have same number of dimensions")
            max_dims = np.max([arr.shape for arr in arrays], axis=0)
            padded_shape = (rows + extra_rows, *max_dims)
            padded = np.full(padded_shape, pad_val, dtype=dtype)
            mask = np.zeros(padded_shape, dtype=bool)

            for i, arr in enumerate(arrays):
                sl = tuple(slice(0, s) for s in arr.shape)
                padded[i][sl] = arr
                mask[i][sl] = True

        if np.all(mask):
            mask = np.array([True], dtype=bool)
        return padded, mask

    def __zero_if_close(self, arr: NDArray | Array, eps_tol_factor=100) -> NDArray | Array:
        tol = eps_tol_factor * np.finfo(arr.dtype).eps
        result = arr.copy()
        result.real[np.abs(result.real) < tol] = 0.0
        result.imag[np.abs(result.imag) < tol] = 0.0
        return result

    def _fft(self, signatures: list[ClauseSignature]) -> tuple[FFT, Array]:
        dfts = []
        idfts = []

        for sig in signatures:
            if sig in self.live_cache:
                # fastest
                clause_fft = self.live_cache[sig]
            elif self.disk_cache and sig in self.disk_cache:
                # check if computed in disk cache
                clause_fft = self.disk_cache[sig]
                self.live_cache[sig] = clause_fft
            else:
                # Compute FFT pair using closed form FWH Coefficients.
                # Longer clauses need more precision, so we are forced to calculate the DFT manually.
                coeffs = self._wf_coeffs(sig)[::-1]
                scale = sig.len + 1
                halfway = scale // 2 + 1
                angles = [-2 * np.pi * Fraction(k, scale) for k in range(1, halfway)]
                unities = [np.cos(angle) + 1j * np.sin(angle) for angle in angles]

                dft = np.ones(scale, dtype=complex)
                dft[1:halfway] = self.__zero_if_close(np.array(unities))
                dft[halfway:] = np.conj(dft[1 : halfway - (sig.len) % 2][::-1])
                dft = dft.reshape(-1, 1)

                idft_powers = np.array([[(i * j) % scale for i in range(scale)] for j in range(scale)], dtype=int)
                idft = np.conjugate(dft[idft_powers].squeeze())
                idft = (coeffs @ idft) / scale
                clause_fft = FFT(dft, idft)
                # update caches
                self.live_cache[sig] = clause_fft
                if self.disk_cache:
                    self.disk_cache.put(sig, clause_fft)

            dfts.append(clause_fft.dft)
            idfts.append(clause_fft.idft)

        dfts, dfts_mask = self._pad(dfts)
        idfts, _ = self._pad(idfts)
        return FFT(dft=jnp.array(dfts), idft=jnp.array(idfts)), jnp.array(dfts_mask)

    def _to_jax_array(self, signatures: list[ClauseSignature], clauses: list[Clauses]) -> ClauseArrays:
        lits, mask = self._pad(clauses)

        if np.all(lits > 0) or np.all(lits < 0):
            sign = np.sign(lits.flatten()[0])
        else:
            sign = np.where(mask, np.sign(lits), 0)
        sign = jnp.array(sign)

        lits = np.where(mask, np.abs(lits) - 1, lits) #adjust for 0 indexing
        lits = jnp.array(lits)
        mask = jnp.atleast_2d(jnp.array(mask))

        types = [clause_type_ids[sig.type] for sig in signatures]
        types, _ = self._pad([[t] for t in types])
        types = jnp.array(types)

        cards = [sig.card for sig in signatures]
        cards, _ = self._pad([[c] for c in cards])
        cards = jnp.array(cards)

        return ClauseArrays(lits, sign, mask, types, cards)

    def process(self, signatures: list[ClauseSignature], clauses: Clauses, benchmark: bool) -> Objective:
        if len(signatures) > 1:
            try:
                assert len(signatures) == len(clauses)
            except AssertionError as e:
                raise (e("Heterogenous clause group must have equal number of clauses and signatures"))

        clauses: ClauseArrays = self._to_jax_array(signatures, clauses)

        ffts, dft_mask = self._fft(signatures)
        # dfts = jnp.broadcast_to(ffts.dft, (clauses.lits.shape[0], max(ffts.dft.shape), 1))
        # ffts = FFT(dft=dfts, idft=ffts.idft)
        if jnp.all(dft_mask) and jnp.all(clauses.mask):
            forward_mask = jnp.array([True], dtype=bool)
        else:
            print("Heterogeneous masks", dft_mask.shape, clauses.mask.shape)
            forward_mask = dft_mask & clauses.mask[:, None, :]
        # TODO:
        # if self.update_diskcache:
        #     self._update_diskcache()
        if not benchmark:
            print("Processed objective has", clauses.lits.shape[0], "clauses with signature(s):", signatures)
        return Objective(clauses=clauses, ffts=ffts, forward_mask=forward_mask)
