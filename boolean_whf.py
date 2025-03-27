import sys
from typing import Optional as Opt, NamedTuple
from abc import ABC, abstractmethod
from jax.typing import ArrayLike as Array

import jax.numpy as jnp
#from jax.experimental.sparse import BCOO
import numpy as np
from scipy.linalg import dft as dft_table
from scipy.special import comb
from scipy.stats import binom


class Clause(NamedTuple):
    type: str
    lits: list[int]
    card: int


class FFT(NamedTuple):
    dft: Array
    idft: Array


class ClauseArrays(NamedTuple):
    lits: Array = jnp.zeros((0,0), dtype=int)
    sign: Array = jnp.zeros((0,0), dtype=int)
    mask: Array = jnp.zeros((0,0), dtype=bool)
    sparse: Array = jnp.zeros((0,0,0), dtype=int)
    weight: Array = jnp.zeros((0,), dtype=float)


class Objective(NamedTuple):
    clauses: ClauseArrays
    ffts: Opt[FFT]
    forward_mask: Opt[Array]
    cards: Opt[Array]


class ClauseGroup(ABC):
    def __init__(self, clauses: list[Clause], n_var: int, do_fft: bool = True, fft_cache: Opt[str] = None):
        self.n_var = n_var
        self.do_fft = do_fft
        self.update_diskcache: bool = False
        if do_fft:
            self.disk_cache: dict = {}
            if fft_cache:
                self.disk_cache = self._load_diskcache(fft_cache)
            self.live_cache: dict = {}
        self.clauses = clauses

    @abstractmethod
    def _wf_coeff(self, t: Opt[str], n: int, k: Opt[int] = None) -> Array:
        pass

    def _load_diskcache(self) -> None:
        # IO object (self.disk_cache) needs to be able to retrieve val on demand from disk
        # e.g. the object loaded in to memory here only knows what keys are on disk
        pass

    def _update_diskcache(self) -> None:
        # load new entries in live cache to disk cache?
        pass

    def _to_array(self, clauses: list[Clause]) -> ClauseArrays:
        lits, card_ks = zip(*[(clause.lits, clause.card) for clause in clauses])
        self.card_ks = jnp.array(card_ks) if all(card_ks) else jnp.zeros((0,0), dtype=int) #.reshape(-1, 1)
        lits, mask = self._pad(lits)
        sign = np.zeros_like(lits)
        np.sign(lits, out=sign, where=mask)
        np.abs(lits, out=lits, where=mask)
        lits[mask] -= 1
        sign = jnp.sign(sign)
        lits = jnp.abs(lits)
        mask = jnp.array(mask)
        weight = jnp.ones(lits.shape[0])

        # Use matrix multiplication instead of indexing
        sp_xlits = jnp.zeros((lits.shape[0], lits.shape[1], self.n_var))
        sp_xlits = sp_xlits.at[jnp.arange(lits.shape[0])[:, None], jnp.arange(lits.shape[1]), lits].set(1)
        #sp_xlits = sparse.BCOO.fromdense(sp_xlits)

        return ClauseArrays(lits, sign, mask, sp_xlits, weight)

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
                self.update_diskcache = True
                # manual compute using closed form FWH Coefficients. 1/n applied to idft.
                dft: Array = dft_table(n + 1)
                coeff = self._wf_coeff(t, n, k)
                print(t, n, k, np.sum(coeff))
                idft = coeff[::-1] @ np.conjugate(dft) / (n + 1)
                dft = dft[1].reshape(n + 1, 1)
                self.live_cache[(t, n, k)] = (dft, idft)

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
            if self.update_diskcache:
                self._update_diskcache()

    def get(self) -> Objective:
        if self.do_fft:
            return Objective(self.clause_array, self.ffts, self.dft_mask, self.card_ks)
        else:
            return Objective(self.clause_array, None, None, self.card_ks)


class CNFClauses(ClauseGroup):
    def __init__(self, clauses: list[Clause], n_var: int, do_fft: bool = True, fft_cache: Opt[str] = None):
        super().__init__(clauses, n_var, do_fft, fft_cache)

    def _wf_coeff(self, t: Opt[str], n: int, k: Opt[int] = None) -> Array:
        d = np.zeros(n + 1)
        d[::1] = [1 / (2 ** (n - 1))] * (n + 1)
        d[0] = d[0] - 1
        return np.array(d)


class EOClauses(ClauseGroup):
    def __init__(self, clauses: list[Clause], n_var: int, do_fft: bool = True, fft_cache: Opt[str] = None):
        super().__init__(clauses, n_var, do_fft, fft_cache)

    def _wf_coeff(self, t: Opt[str], n: int, k: Opt[int] = None) -> Array:
        d = np.zeros(n + 1)
        d[::1] = np.arange(-n, n + 1, 2) / (2 ** (n - 1))  # EO(S) = (2|S|-n)/2^(n-1)
        d[0] = 1 - n / 2 ** (n - 1)  # correct |S| = 0
        return d


class NAEClauses(ClauseGroup):
    def __init__(self, clauses: list[Clause], n_var: int, do_fft: bool = True, fft_cache: Opt[str] = None):
        super().__init__(clauses, n_var, do_fft, fft_cache)

    def _wf_coeff(self, t: Opt[str], n: int, k: Opt[int] = None) -> Array:
        d = np.zeros(n + 1)
        d[::2] = 1 / (2 ** (n - 2))
        d[0] -= 1
        return d


class XORClauses(ClauseGroup):
    def __init__(self, clauses: list[Clause], n_var: int, do_fft: bool = True, fft_cache: Opt[str] = None):
        super().__init__(clauses, n_var, do_fft, fft_cache)

    def _wf_coeff(self, t: Opt[str], n: int, k: Opt[int] = None) -> Array:
        d = np.zeros(n + 1)
        d[-1] = 1
        return d


class AMOClauses(ClauseGroup):
    def __init__(self, clauses: list[Clause], n_var: int, do_fft: bool = True, fft_cache: Opt[str] = None):
        super().__init__(clauses, n_var, do_fft, fft_cache)

    def _wf_coeff(self, t: Opt[str], n: int, k: Opt[int] = None) -> Array:
        d = np.zeros(n + 1)
        d[-1] = 1
        return d


class CardClauses(ClauseGroup):
    def __init__(self, clauses: list[Clause], n_var: int, do_fft: bool = True, fft_cache: Opt[str] = None):
        super().__init__(clauses, n_var, do_fft, fft_cache)

    def _wf_coeff(self, t: Opt[str], n: int, k: Opt[int] = None) -> Array:
        negate = -1 if k < 0 else 1
        k = abs(k)
        if k is None:
            print("ERROR: Cardinality clause detected with no cardinality specified", file=sys.stderr)
            raise TypeError

        # For S where |S| in [0,n], ret[|S|]
        if n == 1 and k == 1:
            print("WARNING: Detected unit literal encoded as cardinality", file=sys.stderr)
            return [0, 1]

        d = np.zeros(n + 1)
        if k == 0:
            print("WARNING: Detected cardinality 0 clause", file=sys.stderr)
            d[0] = -1
            return d

        if k == 1:
            print("WARNING: Detected CNF as cardinality 1 clause", file=sys.stderr)
            d = [1 / (2 ** (n - 1))] * (n + 1)
            d[0] = d[0] - 1
            return np.array(d)

        noise_zeroes = np.array([-1] * (n - k) + [1] * (k - 1))
        noise_poly = np.polynomial.Polynomial.fromroots(noise_zeroes).coef
        if (k - 1) % 2:
            # includes factor of -1 to an odd power - e.g when k is even.
            noise_poly *= -1
        numerators = np.array(comb(n - 1, k - 1, exact=True) * noise_poly)

        m = n - 1  # Convenience
        pascal_row = [1] * (m + 1)
        pascal_row[0] = pascal_row[-1] = 2**m
        pascal = 1
        # Manual computation rather than duplicating work with comb() n times.
        for i in range(1, (m // 2) + 1):
            # i = |S|-1, compute (n-1) choose (|S|-1).
            # n choose (i+1) = n choose i *(n-i)/(i+1)
            pascal = (pascal * (m - i + 1)) // i
            pascal_row[i] = pascal_row[m - i] = pascal * (2**m)
        denominators = np.array(pascal_row)

        d[0] = 2 * binom.cdf(k - 1, n, 0.5) - 1
        d[1:] = negate*(numerators / denominators)
        return d


class ApproxLenClauses(ClauseGroup):
    def __init__(self, clauses: list[Clause], n_var: int, do_fft: bool = True, fft_cache: Opt[str] = None):
        super().__init__(clauses, n_var, do_fft, fft_cache)
        self.walsh_lookup = {cl_type: cl_group([], 0) for cl_type, cl_group in class_map.items()}

    def _wf_coeff(self, t: Opt[str], n: int, k: Opt[int] = None) -> Array:
        return self.walsh_lookup[t]._wf_coeff(t, n, k)


class_map: dict[str, type[ClauseGroup]] = {
    "xor": XORClauses,
    "card": CardClauses,
    "eo": EOClauses,
    "nae": NAEClauses,
    "cnf": CNFClauses,
    "amo": AMOClauses,
}
