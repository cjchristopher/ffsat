from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple
from typing import Optional as Opt

from boolean_whf import Clause, ClauseProcessor, Clauses, ClauseSignature, Objective, clause_type_ids
import numpy as np

logger = logging.getLogger(__name__)


class UnsatError(Exception):
    pass


class PBSATFormula(object):
    def __init__(self, workers: int = 1, n_devices: int = 1, disk_cache: str = None, benchmark: bool = False) -> None:
        self.clause_sets: dict[ClauseSignature, Clauses] = {}
        self.n_var = 0
        self.n_clause = 0
        self.n_devices = n_devices
        self.workers = workers
        self.disk_cache = disk_cache
        self.benchmark = benchmark

    def read_DIMACS(self, dimacs_file: str) -> None:
        """
        Parse SAT problem instances from files in DIMACS format or a hybrid format documented below.
        We assume consistent formatting after the first clause. Mixed format files will not parse correctly.
        This method supports multiple input formats:

        Standard DIMACS CNF -
         - Lines beginning with 'c' or '*' are treated as comments
         - Problem line 'p cnf <vars> <clauses>' specifies the number of variables and clauses
         - Each clause is a space-separated list of integers ending with 0
         - Example: "1 -2 3 0" represents (x₁ ∨ ¬x₂ ∨ x₃)

        Hybrid DIMACS format -
         - As per above with the following changes:
         - All constraint lines either start with ("h" ("h x" ONLY can also be "1 x")) OR the clause type e.g. below.
         - The "h" must be present on all constraints if it is present for any of them.
         Then the following applies:
         - Each constraint begins with a letter (excluding CNF) specifying its type:
           * '' : Standard CNF clause (e.g. "h 1 2 0" or "1 2 0")
           * 'x[or]': XOR (xor) constraint (e.g. "h x .. 0" or "x .. 0", or "xor .. 0", or "1 x .. 0")
           * 'n[ae]': NAE (not-all-equal) constraint (e.g. "h n .. 0" or "n .. 0", or "nae .. 0")
           * 'e[o]': EO (exactly-one) constraint (e.g. "h e .. 0", "h eo .. 0", or "eo .. 0")
           * 'a[mo]': AMO (at-most-one) constraint (e.g. "h a .. 0", "h amo .. 0", or "a .. 0")
           * '[car]d': Cardinality constraint. Cardinality has an additional formatting quirk:
              - Format: "card <k> .. 0" or "h d <k> .. 0" or "d <k> .. 0"
              - Where <k> is a non-zero integer OR has a positive integer with inequality prefix ('<','<=','>','>=')
              - Example: "card 2 1 2 3 0" means "at least 2 of {x₁, x₂, x₃} must be true" ('>=')
              - Example: "card >=2 1 2 3 0" means "at least 2 of {x₁, x₂, x₃} must be true"
              - Example: "card >2 1 2 3 0" means "more than 2 of {x₁, x₂, x₃} must be true"
              - Example: "card -2 1 2 3 0" means "*fewer* than 2 of {x₁, x₂, x₃} must be true" ('<')
              - Example: "card <2 1 2 3 0" means "fewer than 2 of {x₁, x₂, x₃} must be true"
              - Example: "card <=2 1 2 3 0" means "at most 2 of {x₁, x₂, x₃} must be true"

        Args:
            dimacs_file (str): Path to the DIMACS format file to be parsed
        """
        try:
            with open(dimacs_file, "r") as f:
                h_tag = -1  # Hybrid tag specified flag
                for idx, ln in enumerate(f):
                    tokens = ln.split()
                    line = ln.strip()
                    # Skip comments / empties
                    if len(line) == 0 or tokens[0] == "c" or tokens[0] == "*":
                        pass
                    # Problem metadata
                    elif tokens[0] == "p":
                        self.n_var = int(tokens[-2])
                        self.n_clause = int(tokens[-1])
                    # Process contraint
                    else:
                        # TODO: Fix format detection (dimacs vs pbo vs ???).
                        # Could remove this have it check h on every line... would accept mixed format then.
                        # Currently only true once. Will cause files with inconsistent formatting to break atm.
                        if h_tag == -1:
                            if tokens[0] == "h" or (tokens[0] == "1" and tokens[1] == "x"):
                                h_tag = 1
                            else:
                                h_tag = 0

                        clause_type = tokens[h_tag] if tokens[h_tag].isalpha() else "cnf"
                        canon_types = {"d": "card", "x": "xor", "n": "nae", "e": "eo"}
                        clause_type = canon_types[clause_type] if clause_type in canon_types else clause_type
                        if clause_type not in clause_type_ids:
                            logger.warning(f"Unknown clause type: {line}")
                            raise ValueError

                        first_lit = (h_tag + 1) if h_tag else (0 if clause_type == "cnf" else 1)

                        # Cardinality - the next token is cardinality, not a literal.
                        if clause_type == "card":
                            # Catch implicit inequality (is just an int) - e.g. >= card or < card (for negative)
                            try:
                                card = int(tokens[first_lit])
                            # Has an explicit inequality indicator, check for equality and adjust.
                            except ValueError:
                                negate = tokens[first_lit][0] == "<"
                                equality = tokens[first_lit][1] == "="
                                skip = negate + equality
                                card = int(tokens[first_lit][skip:])
                                if card < 0:
                                    logger.error(f"Line {idx}: Inequality cardinality must be positive: {line}")
                                    raise ValueError
                                card = (-1) ** negate * (card + (not (negate ^ equality)))
                            first_lit += 1
                        else:
                            card = 0

                        lits = [int(val) for val in tokens[first_lit:-1]]  # drop trailing 0
                        clause_len = len(lits)

                        # Clause extracted. Check for errors in spec, correct generic edge cases.
                        if clause_len == 0:
                            logger.warning(f"Line {idx}: Skipping empty clause")
                            continue

                        if clause_len == 1 and clause_type != "cnf":
                            if clause_type not in ("amo", "eo", "card"):
                                logger.error(f"Line {idx}: Unit literal with NAE or XOR clause type: {line}")
                                raise ValueError
                            logger.warning(f"Line {idx}: Adjusting unit literal with non-CNF clause type: {line}")
                            clause_type = "cnf"
                            card = 0

                        # Correct CARD edge cases -- flag, correct if possible.
                        if clause_type == "card":
                            if card > clause_len:
                                logger.error(f"Line {idx}: CARD claues with card > #lits (always UNSAT): {line}")
                                raise UnsatError

                            if clause_len == card:
                                n = clause_len
                                logger.warning(f"Line {idx}: Adjusting {n} unit literals enoded as CARD-{n}: {line}")
                                for lit in lits:
                                    sig = ClauseSignature("cnf", clause_len, card)
                                    self.clause_sets.setdefault(sig, []).append(lit)
                                continue

                            if card == 0:
                                logger.warning(f"Line {idx}: Skipping CARD-0 clause (trivially SAT) : {line}")
                                continue

                            if card == 1:
                                logger.warning(f"Line {idx}: Adjusting non-trivial CARD-1 clause to CNF: {line}")
                                clause_type = "cnf"
                                card = 0

                        sig = ClauseSignature(clause_type, clause_len, card)
                        self.clause_sets.setdefault(sig, []).append(lits)

                print(f"Processed file: {dimacs_file}, with {len(self.clause_sets)} objectives (clause sets)"
                      f" - a total {self.n_clause} clauses over {self.n_var} variables "
                      )
        except FileNotFoundError as e:
            print(f"Error: File '{dimacs_file}' not found")
            raise e
        except Exception as e:
            print(f"Error processing file: {e}")
            raise e

    def process_clauses(self) -> tuple[Objective, ...]:
        class Singleton(NamedTuple):
            sig: ClauseSignature
            clause: Clause

        class ClauseGroup(NamedTuple):
            sigs: list[ClauseSignature]
            clauses: Clauses

        clause_grps: list[ClauseGroup] = list()
        singletons_by_len: dict[int, list[Singleton]] = dict()
        padded_group: list[Singleton] = list()

        for clause_sig, clause_list in self.clause_sets.items():
            # Gather singletons by common length for more efficient processing
            if len(clause_list) == 1:
                singletons_by_len.setdefault(clause_sig.clen, []).append(Singleton(clause_sig, clause_list[0]))
            # Homogenous group - add extra dimension to correct processing
            else:
                clause_grps.append(ClauseGroup([clause_sig], clause_list))

        for common_length, singletons in singletons_by_len.items():
            # Collect the unique single lengthers for padded processing
            if len(singletons) == 1:
                padded_group.extend(singletons)
            else:
                sigs = [singleton.sig for singleton in singletons]
                clause_lists: Clauses = [singleton.clause for singleton in singletons]
                clause_grps.append(ClauseGroup(sigs, clause_lists))

        # One last group for all the singletons that don't share a length. This group will be fully padded.
        if padded_group:
            padded_sigs, padded_clause_lists = zip(*padded_group)
            clause_grps.append(ClauseGroup(padded_sigs, padded_clause_lists))

        def parallel_clause_process(clause_grps: list[ClauseGroup], workers: Opt[int] = None) -> list[Objective]:
            processor = ClauseProcessor(self.n_devices, self.disk_cache)

            res: list[Objective] = []
            with ThreadPoolExecutor(max_workers=workers) as tpool:
                tasks = [tpool.submit(processor.process, grp.sigs, grp.clauses, self.benchmark) for grp in clause_grps if grp]
                for task in tasks:
                    res.append(task.result())

            return res

        # N.B. Using multiple works can cause XLA cache conflicts if using "all" persistent caches.
        # All is broken for some jaxopt optimizers however, so we don't use it. If we do, workers should be 1 to avoid
        # race conditions deep in XLA (see jax.config.update("jax_persistent_cache_enable_xla_caches", "all"))
        objectives = parallel_clause_process(clause_grps, workers=min(len(clause_grps), self.workers))
        objectives = tuple(sorted(objectives, key=lambda x: x.clauses.lits.shape[-1]))
        return objectives

    def process_prefix(self, prefix_file: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        try:
            vecs = []
            with open(prefix_file, "r") as f:
                for idx, line in enumerate(f):
                    lits = line.strip().split()
                    if lits[0] in ("c", "#", "*"):
                        continue
                    vec = np.zeros(self.n_var+1, dtype=int)
                    try:
                        lit_vec = np.array([int(lit) for lit in lits], dtype=int)
                        vec[abs(lit_vec[lit_vec<0])] = 1
                        vec[abs(lit_vec[lit_vec>0])] = -1
                        vecs.append(vec.copy())
                    except ValueError:
                        logger.warning(f"Line {idx}: Invalid prefix entry: {line.strip()}")
                        continue
            prefixes = np.delete(np.stack(vecs), 0, axis=1)
            return prefixes
        except FileNotFoundError as e:
            print(f"Error: File '{prefix_file}' not found")
            raise e
        except Exception as e:
            print(f"Error processing prefix file: {e}")
            raise e