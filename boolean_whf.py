import sys
from typing import NamedTuple, Type
from typing import Optional as Opt

import jax.numpy as jnp

# from jax.experimental.sparse import BCOO
import numpy as np
from jax.typing import ArrayLike as Array
from scipy.linalg import dft as dft_table
from scipy.special import comb
from scipy.stats import binom

from diskcache import FFSATCache


class Clause(NamedTuple):
    type: str
    lits: list[int]
    card: int


class FFT(NamedTuple):
    dft: Array
    idft: Array


class ClauseArrays(NamedTuple):
    lits: Array = jnp.zeros((0, 0), dtype=int)
    sign: Array = jnp.zeros((0, 0), dtype=int)
    mask: Array = jnp.zeros((0, 0), dtype=bool)
    weight: Array = jnp.zeros((0,), dtype=float)
    # Any merit to making types negative, and positive numbers indicate card?
    types: Array = jnp.empty((0,), dtype=int)
    cards: Array = jnp.zeros((0, 0), dtype=int)
    sparse: Array = jnp.zeros((0, 0, 0), dtype=int)


class Objective(NamedTuple):
    clauses: ClauseArrays
    ffts: Opt[FFT]
    forward_mask: Opt[Array]
    cards: Opt[Array]


class_idno: dict[str, int] = {"xor": 1, "eo": 2, "nae": 3, "cnf": 4, "amo": 5, "card": 0}


def empty_validator(n_var: int) -> ClauseArrays:
    empty_ClauseArrays = ClauseArrays(sparse=jnp.zeros((0, 0, n_var), dtype=int))
    empty_Validation = Objective(clauses=empty_ClauseArrays, cards = jnp.zeros((0, 0), dtype=int))
    return empty_Validation


class ClauseGroup:
    def __init__(
        self,
        clauses: list[Clause],
        n_var: int,
        do_fft: bool | FFSATCache = True,
        clause_type: Opt[str] = None,
    ):
        self.clauses = clauses
        self.n_var = n_var
        self.do_fft = do_fft
        self.clause_type = clause_type  # Store clause type for diagnostics

        if do_fft:
            self.disk_cache: dict = {}
            if isinstance(do_fft, FFSATCache):
                self.disk_cache: dict = do_fft
            self.live_cache: dict = {}

    def _to_array(self, clauses: list[Clause], sparse_mult: bool = False) -> ClauseArrays:
        # Convert clauses to lists
        lits, cards, types = zip(*[(clause.lits, clause.card, class_idno[clause.type]) for clause in clauses])
        lits, mask = self._pad(lits)

        # Convert to jax
        sign = jnp.where(mask, jnp.sign(lits), 0)
        lits = jnp.where(mask, jnp.abs(lits) - 1, lits)
        types = jnp.array(types)
        cards = jnp.array(cards) if all(cards) else None
        mask = jnp.array(mask)
        weight = jnp.ones(lits.shape[0])

        sparse = None
        if sparse_mult:
            # Use matrix multiplication instead of indexing
            sparse = jnp.zeros((lits.shape[0], lits.shape[1], self.n_var))
            sparse = sparse.at[jnp.arange(lits.shape[0])[:, None], jnp.arange(lits.shape[1]), lits].set(1)
            # sparse = sparse.BCOO.fromdense(sparse)

        return ClauseArrays(lits=lits, sign=sign, mask=mask, weight=weight, types=types, cards=cards, sparse=sparse)

    @classmethod
    def _wf_coeff_cnf(cls, n: int) -> Array:
        """Walsh-Fourier coefficients for CNF clauses."""
        d = [1 / (2 ** (n - 1))] * (n + 1)
        d[0] = d[0] - 1
        return d

    @classmethod
    def _wf_coeff_eo(cls, n: int) -> Array:
        r"""Walsh-Fourier coefficients for exactly-one (EO) clauses.
        $$
        \widehat{\texttt{EO}}(S) = \left\lbrace
        \begin{aligned}
            &1-\frac{n}{2^{n-1}} & \abs{S} = 0 \\
            &\frac{2\abs{S}-n}{2^{n-1}} & \abs{S} \neq 0
        \end{aligned}\right.
        $$"""
        d = np.arange(-n, n + 1, 2) / (2 ** (n - 1))  # EO(S) = (2|S|-n)/2^(n-1)
        d[0] = 1 - n / (2 ** (n - 1))
        return d

    @classmethod
    def _wf_coeff_nae(cls, n: int) -> Array:
        r"""Walsh-Fourier coefficients for not-all-equal (NAE) clauses.
        For all even $$|S|$$, the coefficient at that size is $$(1/2)^{n-2}$$ due to cancellation
        of all even terms in $$g(\rho)$$, and value $${n-1\choose |S|-1}$$ at all odd terms.
        This cancels the denominator in the general equation, leaving a factor of 2.
        This happens to essentially align with the $$|S|=0$$ case, but subtract 1. Hence:
        """
        d = np.zeros(n + 1)
        d[::2] = 1 / (2 ** (n - 2))
        d[0] -= 1
        return d

    @classmethod
    def _wf_coeff_xor(cls, n: int) -> Array:
        """Walsh-Fourier coefficients for XOR clauses.
        All zero expect for the full product of all literals (The nth ESP in n variables).
        """
        d = np.zeros(n + 1)
        d[-1] = 1
        return d

    @classmethod
    def _wf_coeff_amo(cls, n: int) -> Array:
        r"""Walsh-Fourier coefficients for at-most-one (AMO) clauses.
        $$
        \widehat{\texttt{AMO}}(S) = \left\lbrace
        \begin{aligned}
            &1-\frac{n-1}{2^{n-1}} & \abs{S} = 0 \\
            &\frac{2\abs{S}-n-1}{2^{n-1}} & \abs{S} \neq 0
        \end{aligned}\right.
        $$
        """

        d = np.arange(-n - 1, n, 2) / (2 ** (n - 1))
        d[0] = 1 - (n + 1) / (2 ** (n - 1))
        return d

    @classmethod
    def _wf_coeff_card(cls, n: int, k: Opt[int] = None) -> Array:
        r"""Walsh-Fourier coefficients for cardinality clauses.
        This does not simplify to a clean closed form, so we require some manual computation.
        The dominating step is polyfromroots, at $$O(n\cdot log^{2}(n))$$"""
        negate = -1 if k < 0 else 1
        k = abs(k)

        # Check valid $$n,k$$ - inform the user of poor encodings (although this won't affect CLS/FF-Sat solving time)

        # Should never happen.
        if k is None:
            print("ERROR: Cardinality clause detected with no cardinality specified", file=sys.stderr)
            raise TypeError

        # e.g. clause is just CNF.
        if k == 1:
            print("WARNING: Detected CNF encoded as cardinality-1 clause", file=sys.stderr)
            return cls._wf_coeff_cnf(n)

        d = np.zeros(n + 1)
        # e.g. clause $$c = \texttt{CARD}^{\geq n}(x_1, x_2,\ldots, x_n)$$ is just $$FE_c = x_1 \land x_2 \land \ldots \land x_n$$
        # if n == k:
        #     print(f"WARNING: Detected {n} unit literals encoded as a cardinality-{n} clause", file=sys.stderr)
        #     d[-1] = 1
        #     return d

        # e.g. clause $$c = \texttt{CARD}^{\geq 0}(x_1, x_2,\ldots)$$ is just $$FE_c = -1$$ (e.g. free assignment/dead clause)
        if k == 0:
            print("WARNING: Detected cardinality 0 clause", file=sys.stderr)
            d[0] = -1
            return d

        # Calculate $$g(\rho) = (1+\rho)^{n-k}(1-\rho)^{k-1}$$ for given $$n,k$$
        # Reform so all terms in p are zeroes of the poly (nb. $$(1-\rho) = -(\rho-1)$$)
        # $$g(\rho) = (\rho+1)^{n-k}(\rho-1)^{k-1}(-1)^{k-1}$$
        noise_zeroes = np.array([-1] * (n - k) + [1] * (k - 1))
        noise_poly = np.polynomial.polynomial.polyfromroots(noise_zeroes)
        noise_poly *= (-1) ** ((k - 1) % 2)

        # Full numerator expression: $${n-1\choose k-1} g(p)$$
        numerators = np.array(comb(n - 1, k - 1, exact=True) * noise_poly)

        # The denominator for each $$|S|$$ has a $${n-1 \choose |S|-1}$$ term. Since each subsequent term in the series can be expressed
        # as the product of the last term and the current term ($$i$$): $$(n-1)/(i+1)$$, we can save some time computing the
        # m = (n-1)th row of Pascal's triangle (where the top is row 0) progressively (vectorised) vs calling comb().
        m = n - 1
        dtype = np.int64 if m <= 33 else np.object_

        # The triangle is symmetric so we only need the left half of the row.
        pascal_row = np.ones((m // 2) + 1, dtype=dtype)
        pascal_row[1:] = np.arange(1, (m // 2) + 1, dtype=dtype)

        # Calculate the factorial numerator and denominators separately, and integer divide
        # This implements the above recurrence for each subsequent term up to the midpoint of the triangle.
        pascal_denoms = np.cumprod(pascal_row[1:], dtype=dtype)
        np.cumprod((m + 1 - pascal_row[1:]), out=pascal_row[1:], dtype=dtype)
        np.floor_divide(pascal_row[1:], pascal_denoms, out=pascal_row[1:])

        # Flip and concat - offset in case of unique middle element - and finally scale by $$2^{n-1}$$
        offset = (m - 1) % 2
        pascal_row = np.concatenate([pascal_row, pascal_row[::-1][offset:]], dtype=dtype) * 2**m

        d[0] = 2 * binom.cdf(k - 1, n, 0.5) - 1
        d[1:] = negate * (numerators / pascal_row)
        return d

    def _wf_coeff(self, t: str, n: int, k: Opt[int] = None) -> Array:
        """Calculate Walsh-Fourier coefficients for a given clause type."""
        method_map = {
            "cnf": self.__class__._wf_coeff_cnf,
            "eo": self.__class__._wf_coeff_eo,
            "nae": self.__class__._wf_coeff_nae,
            "xor": self.__class__._wf_coeff_xor,
            "amo": self.__class__._wf_coeff_amo,
            "card": self.__class__._wf_coeff_card,
        }

        if t not in method_map:
            raise ValueError(f"Unknown clause type: {t}")

        if t == "card":
            return method_map[t](n, k)
        return method_map[t](n)

    def _fft(self, clauses: list[Clause]) -> tuple[FFT, Array]:
        dfts = []
        idfts = []

        for clause in clauses:
            t = clause.type
            n = len(clause.lits)
            k = clause.card

            if (t, n, k) in self.live_cache:
                # fastest
                dft, idft = self.live_cache[(t, n, k)]
            elif (t, n, k) in self.disk_cache:
                # check if computed in disk cache
                dft, idft = self.disk_cache[(t, n, k)]
                self.live_cache[(t, n, k)] = (dft, idft)
            else:
                # manual compute using closed form FWH Coefficients. 1/n applied to idft.
                dft: Array = dft_table(n + 1)
                coeff = self._wf_coeff(t, n, k)
                idft = coeff[::-1] @ np.conjugate(dft) / (n + 1)
                dft = dft[1].reshape(n + 1, 1)
                self.live_cache[(t, n, k)] = (dft, idft)
                if isinstance(self.do_fft, FFSATCache):
                    self.do_fft.put((t, n, k), (dft, idft))

            dfts.append(dft)
            idfts.append(idft)

        dfts, dfts_mask = [jnp.array(arr) for arr in self._pad(dfts)]
        idfts = jnp.array(self._pad(idfts)[0])
        return (dfts, idfts), dfts_mask

    def _pad(self, arrays: list[Array | list[int]], pad_val: Opt[int] = 0) -> tuple[Array, Array]:
        arrays = [np.array(arr) if isinstance(arr, list) else arr for arr in arrays]
        dtype = np.result_type(*arrays, pad_val)

        if arrays[0].ndim <= 1:
            max_length = max(len(arr) for arr in arrays)
            padded = np.full((len(arrays), max_length), pad_val, dtype=dtype)

            for i, arr in enumerate(arrays):
                padded[i, : len(arr)] = arr

            mask = np.zeros_like(padded, dtype=bool)
            for i, arr in enumerate(arrays):
                mask[i, : len(arr)] = True

        else:
            max_dims = np.max([arr.shape for arr in arrays], axis=0)
            padded_shape = (len(arrays), *max_dims)
            padded = np.full(padded_shape, pad_val, dtype=dtype)
            mask = np.zeros(padded_shape, dtype=bool)

            for i, arr in enumerate(arrays):
                sl = tuple(slice(0, s) for s in arr.shape)
                padded[i][sl] = arr
                mask[i][sl] = True

        return padded, mask

    def process(self) -> None:
        self.clause_array = self._to_array(self.clauses)
        if self.do_fft:
            self.ffts, dft_mask = self._fft(self.clauses)
            # Broadcast extra dim for correct outer addition result shape=(k,n+1,n)
            self.dft_mask = dft_mask & self.clause_array.mask[:, None, :]
            # if self.update_diskcache:
            #     self._update_diskcache()

    def get(self) -> Objective:
        if self.do_fft:
            return Objective(self.clause_array, self.ffts, self.dft_mask, self.card_ks)
        else:
            return Objective(self.clause_array, None, None, self.card_ks)


# Create a function factory for backward compatibility with existing code
def create_clause_group(clause_type: str) -> Type[ClauseGroup]:
    """Factory function that returns a ClauseGroup class specialized for a specific clause type."""

    class SpecializedClauseGroup(ClauseGroup):
        def __init__(self, clauses: list[Clause], n_var: int, do_fft: bool = True, fft_cache: Opt[str] = None):
            super().__init__(clauses, n_var, do_fft, fft_cache, clause_type=clause_type)

    SpecializedClauseGroup.__name__ = f"{clause_type.capitalize()}Clauses"
    return SpecializedClauseGroup
