#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
"""
Quick validation script to check which clauses are unsatisfied by a given solution.

Usage:
    python validate_solution.py <problem_file> <solution_file>

Example:
    python validate_solution.py tests/costas/no_break/costas_9.hybrid solution.sol

The solution file should contain space-separated literals where:
    - Positive integer N means variable N is TRUE
    - Negative integer -N means variable N is FALSE
"""

from __future__ import annotations

import argparse
import sys

import jax.numpy as jnp
import numpy as np
from jax import Array
from boolean_whf import clause_type_ids
from sat_loader import PBSATFormula
from solvers.ffsatsolver import build_eval_verify


def load_solution_file(sol_file: str) -> str:
    """Load solution string from a .sol file."""
    with open(sol_file, "r") as f:
        # Read all lines, skip comments, join remaining content
        lines = []
        for line in f:
            line = line.strip()
            if line and not line.startswith(("#", "c", "*")):
                lines.append(line)
        return " ".join(lines)


def parse_solution(solution_str: str, n_var: int) -> Array:
    """
    Parse a solution string like "-1 2 -3 4 5" into an assignment array.

    Args:
        solution_str: Space-separated literals (positive = true, negative = false)
        n_var: Number of variables in the problem

    Returns:
        Array of shape (n_var + 1,) where index i has value -1 (true) or +1 (false)
        Index 0 is unused (variables are 1-indexed)
    """
    lits = [int(x) for x in solution_str.strip().split()]

    # Assignment array: -1 means satisfied (literal matches), +1 means unsatisfied
    # This matches the convention used in ffsatsolver where sign * x[lits] < 0 means satisfied
    assignment = jnp.zeros(n_var, dtype=jnp.float32)

    for lit in lits:
        var = abs(lit) - 1
        if var > n_var:
            print(f"Warning: Variable {var} exceeds n_var={n_var}, skipping")
            continue
        # Positive lit means var is TRUE -> assignment[var] = -1
        # Negative lit means var is FALSE -> assignment[var] = +1
        assignment = assignment.at[var].set(-1.0 if lit > 0 else 1.0)

    return assignment


def get_unsatisfied_clauses(formula: PBSATFormula, assignment: Array) -> list[dict]:
    """
    Find all clauses unsatisfied by the given assignment.

    Returns a list of dicts with clause information:
        - 'clause_type': str (cnf, xor, eo, etc.)
        - 'literals': list of ints
        - 'cardinality': int (for card/ek clauses)
        - 'objective_idx': which objective group it belongs to
        - 'clause_idx': index within that objective
    """
    objectives = formula.process_clauses_to_array()
    _, verify_fns = build_eval_verify(objectives, unbounded=False)

    unsatisfied = []

    # Reverse lookup for clause type names
    id_to_type = {v: k for k, v in clause_type_ids.items()}

    for obj_idx, (obj, verify_fn) in enumerate(zip(objectives, verify_fns)):
        unsat_mask = verify_fn(assignment)

        unsat_indices = np.where(np.array(unsat_mask.flatten()))[0]

        for clause_idx in unsat_indices:
            clause = set([x for x in (obj.clauses.sign * (obj.clauses.lits + 1))[clause_idx, :].tolist()])
            if len(obj.clauses.types) == 1:
                c_type = obj.clauses.types[0][0]
            else:
                c_type = obj.clauses.types[clause_idx][0]
            if len(obj.clauses.cards) == 1:
                c_card = obj.clauses.cards[0][0]
            else:
                c_card = obj.clauses.cards[clause_idx][0]
            c_type = id_to_type.get(int(c_type), f"unknown({c_type})")
            assign = set([int(-1 * assignment[x] * (abs(x) + 1)) for x in (obj.clauses.lits)[clause_idx, :].tolist()])

            unsatisfied.append(
                {
                    "clause_type": c_type,
                    "cardinality": c_card,
                    "literals": clause,
                    "assigned": assign,
                    "objective_idx": obj_idx,
                    "clause_idx": int(clause_idx),
                }
            )
    return unsatisfied


def format_clause(clause_info: dict) -> str:
    """Format a clause info dict as a human-readable string."""
    ctype = clause_info["clause_type"]
    lits = sorted(list(clause_info["literals"]), key=lambda x: abs(x))
    assignment = sorted(list(clause_info["assigned"]), key=lambda x: abs(x))
    pad = max([len(str(x)) for x in lits] + [len(str(x)) for x in assignment])
    card = clause_info["cardinality"]

    lits_str = " ".join(f"{str(lit):>{pad}}" for lit in lits)
    assn_str = " ".join(f"{str(lit):>{pad}}" for lit in assignment)

    if ctype in ("card", "ek"):
        retstr = f"{ctype.upper():>4}({card:>2}): {lits_str}"
    else:
        retstr = f"{ctype.upper():>6}: {lits_str}"
    retstr += f"\n{' ':>9}: {assn_str}"
    return retstr


def main():
    parser = argparse.ArgumentParser(
        description="Validate a solution against a SAT problem and report unsatisfied clauses"
    )
    parser.add_argument("problem_file", help="Path to the problem file (DIMACS/hybrid format)")
    parser.add_argument("solution_file", help="Path to .sol file containing solution literals")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show satisfied clause count too")

    args = parser.parse_args()

    # Load the problem
    print(f"Loading problem from: {args.problem_file}")
    formula = PBSATFormula(workers=1)
    formula.read_DIMACS(args.problem_file)

    print(f"Problem has {formula.n_var} variables and {formula.n_clause} clauses")

    # Load and parse solution
    print(f"Loading solution from: {args.solution_file}")
    solution_str = load_solution_file(args.solution_file)
    assignment = parse_solution(solution_str, formula.n_var)
    assigned_count = int(jnp.sum(assignment != 0))
    print(f"Solution assigns {assigned_count} variables")

    # Find unsatisfied clauses
    unsatisfied = get_unsatisfied_clauses(formula, assignment)

    # Report results
    if not unsatisfied:
        print("\n✓ All clauses are SATISFIED!")
    else:
        print(f"\n✗ Found {len(unsatisfied)} UNSATISFIED clause(s):\n")
        for i, clause_info in enumerate(unsatisfied, 1):
            print(f"{i}. {format_clause(clause_info)}")
            if args.verbose:
                print(f"      (objective {clause_info['objective_idx']}, clause {clause_info['clause_idx']})")

    if args.verbose:
        total_clauses = formula.n_clause
        sat_count = total_clauses - len(unsatisfied)
        print(f"\nSummary: {sat_count}/{total_clauses} clauses satisfied ({100 * sat_count / total_clauses:.2f}%)")

    return len(unsatisfied)


if __name__ == "__main__":
    sys.exit(main())
