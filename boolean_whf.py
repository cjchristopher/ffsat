from __future__ import annotations

import sys
from typing import NamedTuple
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
    lits: Array = jnp.empty((0, 0), dtype=int)
    sign: Array = jnp.empty((0, 0), dtype=int)
    mask: Array = jnp.empty((0, 0), dtype=bool)
    types: Array = jnp.empty((0, 0), dtype=int)
    cards: Array = jnp.empty((0, 0), dtype=int)
    sparse: Array = jnp.empty((0, 0, 0), dtype=int)


class Objective(NamedTuple):
    clauses: ClauseArrays = ClauseArrays()
    ffts: Opt[FFT] = None
    forward_mask: Opt[Array] = None


class_idno: dict[str, int] = {"xor": 1, "eo": 2, "nae": 3, "cnf": 4, "amo": 5, "card": 0}


def empty_validator(n_var: int) -> ClauseArrays:
    return Objective(clauses=ClauseArrays(sparse=jnp.zeros((0, 0, n_var), dtype=int)))


class ClauseGroup:
    def __init__(
        self,
        clauses: list[Clause],
        n_var: int,
        do_fft: bool | FFSATCache = True,
        clause_type: Opt[str] = None,
        n_devices: int = 1,
    ):
        self.clauses = clauses
        self.n_var = n_var
        self.do_fft = do_fft
        self.clause_type = clause_type  # Store clause type for diagnostics
        self.n_devices = n_devices

        if do_fft:
            self.disk_cache: dict = {}
            if isinstance(do_fft, FFSATCache):
                self.disk_cache: dict = do_fft
            self.live_cache: dict = {}

    def _to_array(self, clauses: list[Clause], sparse_addr: bool = False, n_devices: int = 1) -> ClauseArrays:
        # Convert clauses to lists
        lits, cards, types = zip(*[(clause.lits, [clause.card], class_idno[clause.type]) for clause in clauses])
        lits, mask = self._pad(lits, n_devices = n_devices)
        types, _ = self._pad([[t] for t in types], n_devices = n_devices)
        cards, _ = self._pad(cards, n_devices = n_devices)

        # Convert to jax
        sign = jnp.where(mask, jnp.sign(lits), 0)
        lits = jnp.where(mask, jnp.abs(lits) - 1, lits)
        mask = jnp.array(mask)
        types = jnp.array(types).squeeze()
        cards = jnp.array(cards).squeeze()

        sparse = None
        if sparse_addr:
            # Use matrix multiplication instead of indexing
            sparse = jnp.zeros((lits.shape[0], lits.shape[1], self.n_var))
            sparse = sparse.at[jnp.arange(lits.shape[0])[:, None], jnp.arange(lits.shape[1]), lits].set(1)
            # sparse = sparse.BCOO.fromdense(sparse)

        return ClauseArrays(lits=lits, sign=sign, mask=mask, types=types, cards=cards, sparse=sparse)

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
        For all even $$|S|$$, the coefficient at that size is $$(1/2)^{n-2}$$ due to cancellation of all even terms in $$g(\rho)$$,
        and value $${n-1\choose |S|-1}$$ at all odd terms. This cancels the denominator in the general equation,
        leaving a factor of 2. This happens to essentially align with the $$|S|=0$$ case, but subtract 1. Hence:
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
        The dominating step is polyfromroots, at $$O(n\cdot log^{2}(n))$$ for a clause of $$n$$ variables.
        $$
        \widehat{{\texttt{CARD}_{\geq k}}}(S) = \left\lbrace
        \begin{aligned}
            &1-\frac{\sum_{i=k}^{n}{n\choose i}}{2^{n-1}} & \abs{S} = 0 \\
            &\frac{{n-1 \choose k-1}\left(g_{\texttt{CARD}}(\rho)\right)_{[p^{|S|-1}]}}{{n-1\choose |S|-1}2^{n-1}} & \abs{S} \neq 0
        \end{aligned}\right.
        $$
        and $$ g_{\texttt{CARD}}(\rho) = (1+\rho)^{n-k}(1-\rho)^{k-1} $$
        """
        negate = -1 if k < 0 else 1
        k = abs(k)
        d = np.zeros(n + 1)

        r"""For $$n=k$$, e.g. clause $$c = \texttt{CARD}^{\geq n}(x_1, x_2,\ldots, x_n)$$ is just $$c = x_1 \land x_2 \land \ldots \land x_n$$.
        Then $$g_{\texttt{CARD}} = (1-\rho)^{n-1}$$. The coefficients are just the  $$(n-1)$$th row of Pascal's triangle, 
        with every second term negated, or... $$(-1)^{|S|-1}{n-1 \choose |S|-1}$$, cancelling out.
        This means we get a simplified form:
        $$
        \widehat{{\texttt{CARD}_{\geq k}}}(S) = \left\lbrace
        \begin{aligned}
            &1-\frac{1}{2^{n-1}} & \abs{S} = 0 \\
            &\frac{(-1)^{|S|-1}}{2^{n-1}} & \abs{S} \neq 0
        \end{aligned}\right.
        $$
        Having said this... there is merit to force reincoding this as n separate unit literals.
        Notably those individual terms will be trivially minimisable. This form is distinctly not.
        """
        if n == k:
            d = np.ones(n + 1) / (2 ** (n - 1))
            d[::2] *= -1
            d[0] += 1
            return d

        # # e.g. clause $$c = \texttt{CARD}^{\geq 0}(x_1, x_2,\ldots)$$ is just $$FE_c = -1$$ (always SAT)
        # if k == 0:
        #     d[0] = -1
        #     return d

        # Calculate $$g(\rho) = (1+\rho)^{n-k}(1-\rho)^{k-1}$$, and reformulate for zeroes: $$(1-\rho) = -(\rho-1)$$
        # Therefore $$g(\rho) = (\rho+1)^{n-k}(\rho-1)^{k-1}(-1)^{k-1}$$, and then multiply by combinatoric term.
        noise_zeroes = np.array([-1] * (n - k) + [1] * (k - 1))
        noise_poly = np.polynomial.polynomial.polyfromroots(noise_zeroes)
        noise_poly *= (-1) ** ((k - 1) % 2)
        numerators = np.array(comb(n - 1, k - 1, exact=True) * noise_poly)

        # The denominator for each $$|S|$$ has a $${n-1 \choose |S|-1}$$ term. Since each subsequent term in the series can be expressed
        # as the product of the last term and the current term ($$i$$): $$(n-1)/(i+1)$$, we can save some time computing the
        # m = (n-1)th row of Pascal's triangle (where the top is row 0) progressively (vectorised) vs calling comb().
        m = n - 1
        dtype = np.int64 if m <= 33 else np.object_

        # The triangle is symmetric so we only need the left half of the row.
        pascal_row = np.ones((m // 2) + 1, dtype=dtype)
        pascal_row[1:] = np.arange(1, (m // 2) + 1, dtype=dtype)

        # Calculate the factorial numerator and denominators separately, and integer divide for stability.
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

    def _fft(self, clauses: list[Clause], n_devices: int = 1) -> tuple[FFT, Array, list[Clause]]:
        dfts = []
        idfts = []
        del_list = []
        unit_lits = set()

        for idx, clause in enumerate(clauses):
            t = clause.type
            n = len(clause.lits)
            k = clause.card

            if t == "card":
                # Flag, then correct or discard clauses that shouldn't be CARD.
                if n == k:
                    # flag for deletion, collect unit skips, and skip (handles the $$n=1$$ case)
                    print(f"WARNING: Detected {n} unit literals encoded as a cardinality-{n} clause", file=sys.stderr)
                    del_list.append(idx)
                    for lit in clause.lits:
                        unit_lits.add(lit)
                    continue

                if k == 0:
                    print("WARNING: Detected cardinality 0 clause (trivially SAT)", file=sys.stderr)
                    # flag for deletion and skip
                    del_list.append(idx)
                    continue

                if k == 1:  # implicitly $$n>1$$ at this stage.
                    # Change clause type if we aren't in a "CARD" specific clause group, very minor impact.
                    print("WARNING: Detected CNF encoded as cardinality-1 clause", file=sys.stderr)
                    if self.clause_type != "card":
                        t = "cnf"
                        k = 0
                        clauses[idx] = Clause(type=t, lits=clause.lits, card=k)

            elif n == 1:
                # Single literal clause
                print(f"WARNING: Detected unit literal as {t} clause", file=sys.stderr)
                del_list.append(idx)
                unit_lits.add(clause.lits[0])
                continue

            if (t, n, k) in self.live_cache:
                # fastest
                fft = self.live_cache[(t, n, k)]
            elif (t, n, k) in self.disk_cache:
                # check if computed in disk cache
                fft = self.disk_cache[(t, n, k)]
                self.live_cache[(t, n, k)] = fft
            else:
                # manual compute using closed form FWH Coefficients. 1/n applied to idft.
                coeff = self._wf_coeff(t, n, k)
                dft: Array = dft_table(n + 1)
                idft = coeff[::-1] @ np.conjugate(dft) / (n + 1)
                dft = dft[1].reshape(n + 1, 1)
                fft = FFT(dft, idft)
                # update cachecs
                self.live_cache[(t, n, k)] = fft
                if isinstance(self.do_fft, FFSATCache):
                    self.do_fft.put((t, n, k), fft)

            dfts.append(fft.dft)
            idfts.append(fft.idft)

        for clause_idx in sorted(del_list, reverse=True):
            clauses.pop(clause_idx)

        dfts, dfts_mask = [jnp.array(arr) for arr in self._pad(dfts, n_devices = n_devices)]
        idfts = jnp.array(self._pad(idfts, n_devices = n_devices)[0])
        return fft, dfts_mask, unit_lits

    def _pad(self, arrays: list[Array | list[int]], pad_val: Opt[int] = 0, n_devices: int = 1) -> tuple[Array, Array]:
        arrays = [np.array(arr) if isinstance(arr, (list, int)) else arr for arr in arrays]
        extra_rows = n_devices - (len(arrays) % n_devices)
        dtype = np.result_type(*arrays, pad_val)

        if arrays[0].ndim <= 1:
            max_length = max(len(arr) for arr in arrays)
            padded_shape = (len(arrays)+extra_rows, max_length)
            padded = np.full(padded_shape, pad_val, dtype=dtype)

            for i, arr in enumerate(arrays):
                padded[i, : len(arr)] = arr

            mask = np.zeros_like(padded, dtype=bool)
            for i, arr in enumerate(arrays):
                mask[i, : len(arr)] = True

        else:
            max_dims = np.max([arr.shape for arr in arrays], axis=0)
            padded_shape = (len(arrays)+extra_rows, *max_dims)
            padded = np.full(padded_shape, pad_val, dtype=dtype)
            mask = np.zeros(padded_shape, dtype=bool)

            for i, arr in enumerate(arrays):
                sl = tuple(slice(0, s) for s in arr.shape)
                padded[i][sl] = arr
                mask[i][sl] = True

        return padded, mask

    def process(self) -> None:
        unit_lits = None
        if self.do_fft:
            self.ffts, dft_mask, unit_lits = self._fft(self.clauses, n_devices=self.n_devices)
        self.clause_array = self._to_array(self.clauses, n_devices=self.n_devices)
        if self.do_fft:
            # Get mask. Broadcast extra dim for correct outer addition result shape=(k,n+1,n)
            self.dft_mask = dft_mask & self.clause_array.mask[:, None, :]
            # TODO:
            # if self.update_diskcache:
            #     self._update_diskcache()
        return unit_lits

    def get(self) -> Objective:
        if self.do_fft:
            return Objective(clauses=self.clause_array, ffts=self.ffts, forward_mask=self.dft_mask)
        else:
            return Objective(clauses=self.clause_array)
