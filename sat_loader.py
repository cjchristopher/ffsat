# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from boolean_whf import Clause, ClauseProcessor, Clauses, ClauseSignature, FFSAT_DFTCache, Objective, clause_type_ids

logger = logging.getLogger(__name__)


class UnsatError(Exception):
    pass


class PBSATFormula(object):
    """A class for parsing and processing pseudo-boolean SAT formulas from DIMACS and hybrid formats.
    This class handles loading SAT problem instances from files, processing various constraint types
    (CNF, XOR, NAE, AMO, EO, EK, CARD), and preparing them for parallel computation. It supports
    both standard DIMACS CNF format and an extended hybrid format with additional constraint types.
    Attributes:
        clause_sets (dict[ClauseSignature, Clauses]): Dictionary mapping clause signatures to lists of clauses
        n_var (int): Number of variables in the formula
        n_clause (int): Number of clauses in the formula
        n_devices (int, default=1): Number of CPUs to use for processing
        workers (int, default=1): Number of worker threads for parallel processing
        disk_cache (str): (Optional) Path to disk cache directory for compiled computations
        benchmark (bool): Flag to enable benchmarking mode
    Examples:
        >>> formula = PBSATFormula(workers=4, n_devices=1)
        >>> formula.read_DIMACS("problem.cnf")
        >>> objectives = formula.process_clauses()
        >>> prefixes = formula.process_prefix("prefix.txt")
        - The class supports multiple constraint types beyond standard CNF clauses
        - Parallel processing is used to efficiently handle large problem instances
        - Disk caching can be enabled to avoid recompilation of objectives
        - When using XLA persistent caching with "all" mode, workers should be set to 1
    """

    def __init__(self, workers: int = 1, n_devices: int = 1, disk_cache: str = "", file: str = "") -> None:
        self.clause_sets: dict[ClauseSignature, Clauses] = {}
        self.n_var: int = 0
        self.n_clause: int = 0
        self.n_devices: int = n_devices
        self.workers: int = workers
        self.unit_prefix: set[int] = set()
        self.disk_cache: FFSAT_DFTCache | None = None
        if disk_cache:
            self.disk_cache = FFSAT_DFTCache(disk_cache)

        if file:
            self.read_DIMACS(file)

    def read_DIMACS(self, dimacs_file: str) -> None:
        """
        Parse SAT problem instances from files in DIMACS format or a hybrid format documented below.
        We assume consistent formatting after the first clause. Mixed format files will not parse correctly.
        This method supports multiple input formats:

        Standard DIMACS CNF -
         - Lines beginning with 'c' or '*' are treated as comments
         - Problem line 'p cnf <vars> <clauses>' specifies the number of variables and clauses
         - Each clause is a space-separated list of integers ending with 0
         - Example: "1 -2 3 0" represents ($x_1 \vee \neg x_2 \vee x_3$)

        Hybrid DIMACS format -
         - As per above with the following changes:
         - All constraint lines either start with "h" (n.b. "h x" ONLY can also be "1 x") OR the clause type e.g. below.
         - The "h" must be present on all constraints if it is present for any of them.
         Then the following applies:
         - Each constraint begins with a letter (excluding CNF) specifying its type:
           * '' : Standard CNF clause (e.g. "h 1 2 0" or "1 2 0")
           * 'x[or]': XOR (xor) constraint (e.g. "h x .. 0" or "x .. 0", or "xor .. 0", or "1 x .. 0")
           * 'n[ae]': NAE (not-all-equal) constraint (e.g. "h n .. 0" or "n .. 0", or "nae .. 0")
           * 'a[mo]': AMO (at-most-one) constraint (e.g. "h a .. 0", "h amo .. 0", or "a .. 0")
           * 'e[o]': EO (exactly-one) constraint (e.g. "h e .. 0", "h eo .. 0", or "eo .. 0")
           * 'k|ek': EK (exactly-K) constraint. EK has an additional formatting quirk:
              - Format: "k <k> .. 0" or "h k <k> .. 0" or "h ek <k> .. 0"
              - Where <k> is a positive integer
              - Example: "ek 2 1 2 3 0" means "exactly 2 of {$x_1, x_2, x_3$} must be true"
           * '[car]d': CARD (cardinality) constraint. CARD has an additional formatting quirk:
              - Format: "card <k> .. 0" or "h d <k> .. 0" or "d <k> .. 0"
              - Where <k> is a non-zero integer OR has a positive integer with inequality prefix ('<','<=','>','>=')
              - If no inequality prefix is supplied, then:
                - For positive k the default is '>=' (at least k true)
                - For negative k the default is '<' (at most k-1 true)
              - Example: "card 2 1 2 3 0" means "at least 2 of {$x_1, x_2, x_3$} must be true" ('>=')
              - Example: "card >=2 1 2 3 0" means "at least 2 of {$x_1, x_2, x_3$} must be true"
              - Example: "card >2 1 2 3 0" means "more than 2 of {$x_1, x_2, x_3$} must be true"
              - Example: "card -3 1 2 3 4 0" means "*fewer* than 3 of {$x_1, x_2, x_3, x_4$} must be true" ('<')
              - Example: "card <3 1 2 3 4 0" means "fewer than 3 of {$x_1, x_2, x_3, x_4$} must be true"
              - Example: "card <=2 1 2 3 0" means "at most 2 of {$x_1, x_2, x_3$} must be true"

        Args:
            dimacs_file (str): Path to the DIMACS format file to be parsed
        """

        def __process_clause(idx: int, line: str, tokens: list[str]) -> None:
            """
            Helper function to processes and validates a single clause, updating self.clause_sets
            Args:
                idx (int): Line number in the file for error reporting.
                line (str): The original line text for error messages.
                tokens (list[str]): Tokenized clause components including optional clause type,
                                   cardinality value (for card/ek clauses), literals, and trailing zero.
            Raises:
                ValueError: If the clause is malformed, has unknown type, or violates clause semantics.
                UnsatError: If the clause or its implications create an immediate unsatisfiability
                           (e.g., conflicting unit literals, card > n).
            """
            h_offset = 0
            if tokens[0] == "h" or (tokens[0] == "1" and tokens[1] == "x"):  # "1 x" valid when using "h"
                h_offset = 1

            clause_type = tokens[h_offset] if tokens[h_offset].isalpha() else "cnf"
            canon_types = {"d": "card", "x": "xor", "n": "nae", "e": "eo"}
            clause_type = canon_types[clause_type] if clause_type in canon_types else clause_type
            if clause_type not in clause_type_ids:
                logger.error(f"Unknown clause type: {line}")
                raise ValueError

            # First part of clause is always after the (optional) h and clause type (unless cnf)
            first_lit_or_card = h_offset if clause_type == "cnf" else (h_offset + 1)

            if first_lit_or_card >= len(tokens) - 1:
                logger.error(f"Line {idx}: Malformed clause: {line}")
                raise ValueError

            if clause_type in ("card", "ek"):
                # Catch implicit inequality (is just an int) - e.g. >= card or < card (for negative)
                try:
                    card = int(tokens[first_lit_or_card])

                # Must have an explicit inequality indicator or is malformed/typo
                except ValueError:
                    # Check for strict inequality. All cardinality clauses are normalised to default forms:
                    # ">=" or "<", so we adjust the sign and value of k (card).
                    ineq = tokens[first_lit_or_card]
                    if ineq[0] in ("<", ">") and clause_type == "card":
                        negate = ineq[0] == "<"
                        equality = ineq[1] == "="
                        skip = negate + equality
                        card = int(ineq[skip:])
                        if card < 0:
                            logger.error(f"Line {idx}: Inequality cardinality must be positive: {line}")
                            raise ValueError
                        card = (-1) ** negate * (card + (not (negate ^ equality)))
                    else:
                        logger.error(f"Line {idx}: EK constraint cannot have inequality: {line}")
                        raise ValueError
                first_lit_or_card += 1
            else:
                card = 0

            lits = [int(val) for val in tokens[first_lit_or_card:-1]]  # drop trailing 0
            n = len(lits)

            # Clause extracted. Check for errors in spec, correct generic edge cases.
            if n == 0:
                logger.warning(f"Line {idx}: Skipping empty clause")
                return

            if n == 1:
                match clause_type:
                    case "nae" | "xor":
                        logger.error(f"Line {idx}: Length 1 NAE/XOR clause type has no semantics: {line}")
                        raise ValueError

                    case "amo":
                        logger.warning(f"Line {idx}: Skipping length 1 AMO clause (trivially SAT): {line}")
                        return

                    case "eo" | "cnf":
                        logger.warning(f"Line {idx}: Prefixing unit literals enoded as EO: {line}")
                        if -lits[0] in self.unit_prefix:
                            logger.error(f"Conflict found among unit literals - {dimacs_file} is UNSAT")
                            raise UnsatError
                        else:
                            self.unit_prefix.add(lits[0])
                            return
                    case _:
                        pass

            # Correct CARD/EK edge cases -- flag, correct if possible.
            if clause_type in ("card", "ek"):
                if card > n:
                    logger.error(f"Line {idx}: CARD/EK claues with card > #lits (always UNSAT): {line}")
                    raise UnsatError

                if card == n:
                    logger.warning(f"Line {idx}: Prefixing {n} unit literals enoded as CARD/EK-{n}: {line}")
                    for lit in lits:
                        if -lit in self.unit_prefix:
                            logger.error(f"Conflict found among unit literals - {dimacs_file} is UNSAT")
                            raise UnsatError
                        else:
                            self.unit_prefix.add(lit)
                            return

                if card == 0:
                    if clause_type == "card":
                        logger.warning(f"Line {idx}: Skipping CARD-0 clause (trivially SAT): {line}")
                        return
                    else:
                        logger.warning(f"Line {idx}: Prefixing negated EK-0 clause (trivially SAT): {line}")
                        for lit in lits:
                            if lit in self.unit_prefix:
                                logger.error(f"Conflict found among unit literals - {dimacs_file} is UNSAT")
                                raise UnsatError
                            else:
                                self.unit_prefix.add(-lit)

                if card == 1:
                    if clause_type == "card":
                        logger.warning(f"Line {idx}: Adjusting non-trivial CARD-1 clause to CNF: {line}")
                        clause_type = "cnf"
                    else:
                        logger.warning(f"Line {idx}: Adjusting non-trivial EK-1 clause to EO: {line}")
                        clause_type = "eo"
                    card = 0

            self.clause_sets.setdefault(ClauseSignature(clause_type, n, card), []).append(lits)

        try:
            with open(dimacs_file, "r") as f:
                self.h_offset = -1  # Reset flag
                for idx, ln in enumerate(f):
                    tokens = ln.split()
                    line = ln.strip()

                    # Skip comments / empties
                    if len(line) == 0 or tokens[0] == "c" or tokens[0] == "*":
                        pass

                    # Problem metadata
                    elif tokens[0] == "p":
                        print(len(tokens), tokens)
                        if len(tokens) == 3 or len(tokens) == 4:
                            self.n_var = int(tokens[-2])
                            self.n_clause = int(tokens[-1])
                        else:
                            logger.error(f"Line {idx}: Malformed problem specification: {line}")
                            raise ValueError

                    # Process contraint
                    else:
                        if len(tokens) < 2 or tokens[-1] != "0":
                            logger.error(f"Line {idx}: Malformed clause: {line}")
                            raise ValueError
                        __process_clause(idx, line, tokens)

                print(
                    f"Processed file: {dimacs_file}, with {len(self.clause_sets)} objectives (clause sets)"
                    f" - a total of {self.n_clause} clauses over {self.n_var} variables"
                )
        except FileNotFoundError as e:
            print(f"Error: File '{dimacs_file}' not found")
            raise e
        except Exception as e:
            print(f"Error processing file: {e}")
            raise e

    def process_clauses_to_array(self) -> tuple[Objective, ...]:
        """
        Process and group clauses for efficient parallel computation.
        This method organizes clauses into groups based on their signatures and lengths,
        then processes them in parallel to create Objective instances.
        Grouping strategy:
            - Singletons (unique clause signatures): Grouped into single array by clause length
            - Non-singletons (multiple clauses per signature): Kept as separate arrays
            - Unique-length singletons: Combined into a single padded group
        The method uses multithreading to process clause groups in parallel, with each
        group being transformed into an Objective by the ClauseProcessor.
        Returns:
            tuple[Objective, ...]: A sorted tuple of Objective instances, ordered by
                the number of literals in their clauses (ascending).
        Notes:
            - Using multiple workers with XLA's "all" persistent cache mode can cause
              cache conflicts. If persistent caching is enabled for all components,
              workers should be set to 1 to avoid race conditions.
            - The number of workers is capped at the minimum of clause groups count
              and the configured worker limit.
        """

        class Singleton(NamedTuple):
            # A singleton is a clause that is unique in its overall signature for the problem
            sig: ClauseSignature
            clause: Clause

        class ClauseGroup(NamedTuple):
            sigs: list[ClauseSignature]
            clauses: Clauses

        clause_grps: list[ClauseGroup] = list()
        singletons_by_len: dict[int, list[Singleton]] = dict()
        padded_group: list[Singleton] = list()

        for set_signature, set_clauses in self.clause_sets.items():
            # Gather singletons by common length for more efficient processing
            if len(set_clauses) == 1:
                singletons_by_len.setdefault(set_signature.len, []).append(Singleton(set_signature, set_clauses[0]))

            # Homogenous set, so no grouping required
            else:
                clause_grps.append(ClauseGroup([set_signature], set_clauses))

        for singletons in singletons_by_len.values():
            # Collect the unique single lengthers for padded processing
            if len(singletons) == 1:
                padded_group.extend(singletons)
            else:
                sigs, clause_lists = zip(*singletons)
                clause_grps.append(ClauseGroup(list(sigs), list(clause_lists)))

        if padded_group:
            padded_sigs, padded_clause_lists = zip(*padded_group)
            clause_grps.append(ClauseGroup(list(padded_sigs), list(padded_clause_lists)))

        def parallel_clause_process(clause_grps: list[ClauseGroup], workers: int = 1) -> list[Objective]:
            processor = ClauseProcessor(self.n_devices, self.disk_cache)

            res: list[Objective] = []
            with ThreadPoolExecutor(max_workers=workers) as tpool:
                tasks = [tpool.submit(processor.process, grp.sigs, grp.clauses) for grp in clause_grps if grp]
                for task in tasks:
                    res.append(task.result())

            return res

        # N.B. Using multiple workers can cause XLA cache conflicts if using "all" persistent caches.
        # All is broken for some jaxopt optimizers however, so we don't use it. If we do, workers should be 1 to avoid
        # race conditions deep in XLA (see jax.config.update("jax_persistent_cache_enable_xla_caches", "all"))
        objectives = parallel_clause_process(clause_grps, workers=min(len(clause_grps), self.workers))
        objectives = tuple(sorted(objectives, key=lambda x: x.clauses.lits.shape[-1]))
        return objectives

    def process_prefix(self, prefix_file: str) -> NDArray:
        def __lits_to_prefix(lits: Iterable[int]) -> NDArray:
            vec = np.zeros(self.n_var + 1, dtype=int)
            lit_vec = np.array([int(lit) for lit in lits], dtype=int)
            vec[abs(lit_vec[lit_vec < 0])] = 1
            vec[abs(lit_vec[lit_vec > 0])] = -1
            return vec

        try:
            vecs = []
            if self.unit_prefix:
                vecs.append(__lits_to_prefix(self.unit_prefix))
            with open(prefix_file, "r") as f:
                for idx, line in enumerate(f):
                    lits = line.strip().split()
                    if lits[0] in ("c", "#", "*"):
                        continue
                    try:
                        lits = set(-int(lit) for lit in lits)
                        conflict = self.unit_prefix.intersection(lits)
                        if conflict:
                            logger.error(f"Conflict ({conflict}) found among unit literals with prefix-{idx} - {line}")
                            raise UnsatError
                        vecs.append(__lits_to_prefix(lits))
                    except Exception:
                        logger.warning(f"Line {idx}: Invalid prefix entry: {line.strip()}")
                        continue
            prefixes = np.delete(np.stack(vecs), 0, axis=1)  # purge leading zeros.
            return prefixes

        except FileNotFoundError as e:
            print(f"Error: File '{prefix_file}' not found")
            raise e

        except Exception as e:
            print(f"Error processing prefix file: {e}")
            raise e
