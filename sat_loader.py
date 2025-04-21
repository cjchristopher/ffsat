import sys
from boolean_whf import Clause, class_idno


class Formula(object):
    def __init__(self):
        self.clauses_type: dict[str, list[Clause]] = {clause_class: [] for clause_class in class_idno}
        self.clauses_len: dict[int, list[Clause]] = {}
        self.clauses_all: list[Clause] = []
        self.stats = {clause_class: {} for clause_class in class_idno}
        self.n_var = 0
        self.n_clause = 0

    def read_DIMACS(self, dimacs_file: str):
        """
        Parse SAT problem instances from files in DIMACS format or a hybrid format documented below.

        This method supports multiple input formats:

        Standard DIMACS CNF:
         - Lines beginning with 'c' or '*' are treated as comments
         - Problem line 'p cnf <vars> <clauses>' specifies the number of variables and clauses
         - Each clause is a space-separated list of integers ending with 0
         - Example: "1 -2 3 0" represents (x₁ ∨ ¬x₂ ∨ x₃)

        Hybrid DIMACS format:
         - As per above with the following changes:
         - All constraint lines either start with ("h" (with "1 x" valid as a sub for "h x")) OR the clause type -
         Then the following applies:
         - Each constraint begins with a letter (excluding CNF) specifying its type:
           * '' : Standard CNF clause (e.g. "h 1 2 0" or "1 2 0")
           * 'x[or]': XOR (xor) constraint (e.g. "h x .. 0" or "x .. 0", or "xor .. 0")
           * 'n[ae]': NAE (not-all-equal) constraint (e.g. "h n .. 0" or "n .. 0", or "nae .. 0")
           * 'e[o]': EO (exactly-one) constraint (e.g. "h e .. 0", "h eo .. 0", or "eo .. 0")
           * 'a[mo]': AMO (at-most-one) constraint (e.g. "h a .. 0", "h amo .. 0", or "a .. 0")
           * 'd | card': Cardinality constraint. Cardinality has an additional formatting quirk:
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

        Returns:
            int: 0 on success, 1 on error (file not found or parsing error)

        Side effects:
            Populates self.clauses_type, self.clauses_len, self.clauses_all, self.stats,
            self.n_var, and self.n_clause with the parsed problem data

        Note:
            The function assumes consistent formatting throughout the file. Files with
            mixed format styles may not parse correctly.
        """
        try:
            with open(dimacs_file, "r") as f:
                h_tag = -1
                for line in f:
                    split = line.split()
                    if len(line) == 0 or split[0] == "c" or split[0] == "*":
                        # Comments
                        pass
                    elif split[0] == "p":
                        # Read p line (problem metadata)
                        self.n_var = int(split[-2])
                        self.n_clause = int(split[-1])
                    else:
                        # Process constraint
                        # TODO: Fix format detection (dimacs vs pbo vs ???)
                        if h_tag == -1:  # only true once. Will cause files with inconsistent formatting to break
                            if split[0] == "h" or (split[0] == "1" and split[1] == "x"):
                                h_tag = 1
                            else:
                                h_tag = 0

                        # Get clause type, rename for clarity, check validity.
                        clause_type = split[h_tag] if split[h_tag].isalpha() else "cnf"
                        type_map = {"d": "card", "x": "xor", "n": "nae", "e": "eo"}
                        clause_type = type_map[clause_type] if clause_type in type_map else clause_type
                        if clause_type not in self.clauses_type:
                            print(f"WARNING: Unknown clause type: {line}", file=sys.stderr)
                            raise ValueError

                        first = (h_tag + 1) if h_tag else (0 if clause_type == "cnf" else 1)

                        if clause_type == "card":  # Cardinality - extract k as the first number
                            try:
                                # Is just an int, e.g. >= card or < card (for negative)
                                card = int(split[first])
                            except Exception:
                                # Has an explicit inequality indicator, check for = and adjust.
                                negate = True if split[first][0] == "<" else False
                                equality = True if split[first][1] == '=' else False
                                skip = negate + equality
                                card = int(split[first][skip:])
                                if card < 0:
                                    print("Cardinality clause can't be an inequality against a negative cardinality")
                                    raise ValueError
                                card = (-1)**negate * (card + (not(negate^equality)))
                            first += 1  # new first literal position
                        else:
                            card = 0

                        # Collect literals
                        lits = [int(val) for val in split[first:-1]]  # drop trailing 0
                        clause = Clause(clause_type, lits, card)
                        n = len(clause.lits)

                        # Append clauses to relevant lists
                        self.clauses_type[clause_type].append(clause)
                        if n in self.clauses_len:
                            self.clauses_len[n].append(clause)
                        else:
                            self.clauses_len[n] = [clause]
                        self.clauses_all.append(clause)

                        if n in self.stats[clause_type]:
                            self.stats[clause_type][n] += 1
                        else:
                            self.stats[clause_type][n] = 1
                print(f"Successfully processed file: {dimacs_file}")
        except FileNotFoundError as e:
            print(f"Error: File '{dimacs_file}' not found")
            raise e
        except Exception as e:
            print(f"Error processing file: {e}")
            raise e
