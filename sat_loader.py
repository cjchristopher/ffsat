import sys
from boolean_whf import Clause, class_map


class Formula(object):
    def __init__(self):
        self.clauses_type: dict[str, list[Clause]] = {clause_class: [] for clause_class in class_map}
        self.clauses_len: dict[int, list[Clause]] = {}
        self.clauses_all: list[Clause] = []
        self.stats = {clause_class: {} for clause_class in class_map}
        self.n_var = 0
        self.n_clause = 0

    def read_DIMACS(self, dimacs_file: str):
        try:
            with open(dimacs_file, "r") as f:
                h_tag = -1
                for line in f:
                    split = line.split()
                    if len(line) == 0 or split[0] == "c" or split[0] == "*":
                        # comments
                        pass
                    elif split[0] == "p":
                        # Read p line (problem metadata)
                        self.n_var = int(split[-2])
                        self.n_clause = int(split[-1])
                    else:
                        # line is a constraint
                        # TODO: Fix format detection (dimacs vs pbo vs ???)
                        # Check for dimacs vs modified(hybrid) dimacs/pbo
                        if h_tag == -1:  # gate that is hopefully optimisable
                            if split[0] == "h" or (split[0] == "1" and split[1] == "x"):
                                h_tag = 1
                            else:
                                h_tag = 0

                        # Get clause type, rename for clarity, check validity.
                        clause_type = split[h_tag] if split[h_tag].isalpha() else "cnf"
                        type_map = {"d": "card", "x": "xor", "n": "nae"}
                        clause_type = type_map[clause_type] if clause_type in type_map else clause_type
                        if clause_type not in self.clauses_type:
                            print(f"WARNING: Unknown clause type: {line}", file=sys.stderr)
                            raise ValueError

                        # Collect literals
                        first = (h_tag + 1) if h_tag else (0 if clause_type == "cnf" else 1)
                        lits = [int(val) for val in split[first:-1]]  # drop trailing 0

                        if clause_type == "card":  # Cardinality - extract k.
                            clause = Clause(clause_type, lits[1:], lits[0])
                        else:
                            clause = Clause(clause_type, lits, None)

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
        except FileNotFoundError:
            print(f"Error: File '{dimacs_file}' not found")
            return 1
        except Exception as e:
            print(f"Error processing file: {e}")
            return 1
