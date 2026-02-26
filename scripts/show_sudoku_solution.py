#!/usr/bin/python3

"""
Sudoku solution visualizer.

Takes a variable assignment (from a SAT solver) and displays the resulting
sudoku grid, highlighting any conflicts.

Variable encoding (from gen.py):
  var(r, c, v) = (r-1)*N*N + (c-1)*N + (v-1) + 1
  where r, c, v are 1-indexed and N = D*D (default D=3, N=9).

Usage:
  python show_solution.py <assignment_file> [--dim D]

Assignment file formats supported:
  - One literal per line (positive = true, negative = false)
  - Space-separated literals on one line (DIMACS solution format)
  - Lines starting with 'v ' (minisat-style output)
  - Lines starting with 's ' are skipped (status lines)
  - Lines starting with 'c ' are skipped (comments)
"""

import sys
import argparse

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def decode_var(literal: int, N: int) -> tuple[int, int, int]:
    """Decode a variable number back to (r, c, v), all 1-indexed."""
    v_id = abs(literal) - 1  # 0-indexed
    v = (v_id % N) + 1
    v_id //= N
    c = (v_id % N) + 1
    r = (v_id // N) + 1
    return r, c, v


def var(r: int, c: int, v: int, N: int) -> int:
    """Encode (r, c, v) to a variable number. All 1-indexed."""
    return (r - 1) * N * N + (c - 1) * N + (v - 1) + 1


def parse_assignment(filename: str) -> set[int]:
    """Parse an assignment file and return a set of true literals."""
    true_lits = set()
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c ") or line.startswith("s "):
                continue
            # Strip 'v ' prefix (minisat output format)
            if line.startswith("v "):
                line = line[2:]
            tokens = line.split()
            for tok in tokens:
                try:
                    lit = int(tok)
                    if lit == 0:
                        continue  # End-of-clause marker
                    if lit > 0:
                        true_lits.add(lit)
                except ValueError:
                    continue
    return true_lits


def build_grid(true_lits: set[int], N: int) -> list[list[list[int]]]:
    """
    Build a grid from the true literals.
    Returns grid[r][c] = list of values assigned to that cell (ideally length 1).
    Indices are 0-based.
    """
    grid: list[list[list[int]]] = [[[] for _ in range(N)] for _ in range(N)]
    max_var = N * N * N
    for lit in true_lits:
        if lit < 1 or lit > max_var:
            continue
        r, c, v = decode_var(lit, N)
        if 1 <= r <= N and 1 <= c <= N and 1 <= v <= N:
            grid[r - 1][c - 1].append(v)
    # Sort values in each cell
    for r in range(N):
        for c in range(N):
            grid[r][c].sort()
    return grid


def find_conflicts(grid: list[list[list[int]]], D: int) -> set[tuple[int, int]]:
    """
    Find all cells involved in conflicts. Returns set of (r, c) 0-indexed pairs.

    Conflicts:
      - Cell has 0 values (unassigned) or >1 values (multi-assigned)
      - Same value appears twice in a row, column, or subgrid
    """
    N = D * D
    conflicts: set[tuple[int, int]] = set()

    for r in range(N):
        for c in range(N):
            if len(grid[r][c]) != 1:
                conflicts.add((r, c))

    # Row conflicts
    for r in range(N):
        seen_row: dict[int, list[int]] = {}
        for c in range(N):
            for v in grid[r][c]:
                seen_row.setdefault(v, []).append(c)
        for v, cols in seen_row.items():
            if len(cols) > 1:
                for c in cols:
                    conflicts.add((r, c))

    # Column conflicts
    for c in range(N):
        seen_col: dict[int, list[int]] = {}
        for r in range(N):
            for v in grid[r][c]:
                seen_col.setdefault(v, []).append(r)
        for v, rows in seen_col.items():
            if len(rows) > 1:
                for r in rows:
                    conflicts.add((r, c))

    # Subgrid conflicts
    for sr in range(D):
        for sc in range(D):
            seen: dict[int, list[tuple[int, int]]] = {}
            for dr in range(D):
                for dc in range(D):
                    r, c = sr * D + dr, sc * D + dc
                    for v in grid[r][c]:
                        seen.setdefault(v, []).append((r, c))
            for v, cells in seen.items():
                if len(cells) > 1:
                    for cell in cells:
                        conflicts.add(cell)

    return conflicts


def print_grid(grid: list[list[list[int]]], conflicts: set[tuple[int, int]], D: int):
    """Pretty-print the sudoku grid with conflict highlighting."""
    N = D * D
    cell_width = 2 if N <= 9 else 3

    def h_sep():
        parts = []
        for sb in range(D):
            parts.append("-" * (D * cell_width + D - 1))
        return "+" + "+".join(parts) + "+"

    print()
    for r in range(N):
        if r % D == 0:
            print(h_sep())
        row_parts = []
        for c in range(N):
            vals = grid[r][c]
            if len(vals) == 0:
                cell_str = ".".rjust(cell_width)
            elif len(vals) == 1:
                cell_str = str(vals[0]).rjust(cell_width)
            else:
                cell_str = "+".rjust(cell_width)
                #cell_str = (",".join(str(v) for v in vals))[:cell_width].rjust(cell_width)

            if (r, c) in conflicts:
                cell_str = RED + BOLD + cell_str + RESET
            else:
                cell_str = GREEN + cell_str + RESET

            row_parts.append(cell_str)

        # Group into subgrid blocks
        line = "|"
        for sb in range(D):
            block = row_parts[sb * D : (sb + 1) * D]
            line += " ".join(block) + "|"
        print(line)
    print(h_sep())


def print_summary(grid: list[list[list[int]]], conflicts: set[tuple[int, int]], D: int):
    """Print a summary of the solution quality."""
    N = D * D
    total_cells = N * N
    assigned = sum(1 for r in range(N) for c in range(N) if len(grid[r][c]) == 1)
    unassigned = sum(1 for r in range(N) for c in range(N) if len(grid[r][c]) == 0)
    multi = sum(1 for r in range(N) for c in range(N) if len(grid[r][c]) > 1)

    print()
    print(f"Cells assigned:    {assigned}/{total_cells}")
    if unassigned > 0:
        print(f"{YELLOW}Cells unassigned:  {unassigned}{RESET}")
    if multi > 0:
        print(f"{RED}Cells multi-value: {multi}{RESET}")

    if conflicts:
        print(f"{RED}{BOLD}Conflicts found:   {len(conflicts)} cells involved{RESET}")
        print()
        print(f"{RED}Conflict details:{RESET}")
        # Categorize conflicts
        for r, c in sorted(conflicts):
            vals = grid[r][c]
            if len(vals) == 0:
                print(f"  ({r+1},{c+1}): unassigned")
            elif len(vals) > 1:
                print(f"  ({r+1},{c+1}): multiple values {vals}")
            else:
                # Find what it conflicts with
                reasons = []
                v = vals[0]
                # Row
                row_dupes = [c2 + 1 for c2 in range(N) if c2 != c and v in grid[r][c2]]
                if row_dupes:
                    reasons.append(f"row {r+1} (also at col {row_dupes})")
                # Col
                col_dupes = [r2 + 1 for r2 in range(N) if r2 != r and v in grid[r2][c]]
                if col_dupes:
                    reasons.append(f"col {c+1} (also at row {col_dupes})")
                # Subgrid
                sr, sc = (r // D) * D, (c // D) * D
                sg_dupes = []
                for dr in range(D):
                    for dc in range(D):
                        r2, c2 = sr + dr, sc + dc
                        if (r2, c2) != (r, c) and v in grid[r2][c2]:
                            sg_dupes.append((r2 + 1, c2 + 1))
                if sg_dupes:
                    reasons.append(f"subgrid (also at {sg_dupes})")
                print(f"  ({r+1},{c+1}): value {v} conflicts in {'; '.join(reasons)}")
    else:
        print(f"{GREEN}{BOLD}No conflicts -- valid solution!{RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a SAT solver sudoku assignment and check for conflicts."
    )
    parser.add_argument("assignment", help="File containing the variable assignment (literals)")
    parser.add_argument(
        "--dim", "-d", type=int, default=3,
        help="Subgrid dimension D (grid is DxD by DxD, default: 3 for standard 9x9)"
    )
    args = parser.parse_args()

    D = args.dim
    N = D * D

    print(f"Sudoku dimension: {N}x{N} (subgrid {D}x{D})")
    print(f"Variables: 1..{N*N*N}")

    true_lits = parse_assignment(args.assignment)
    relevant = {lit for lit in true_lits if 1 <= lit <= N * N * N}
    print(f"True literals in range: {len(relevant)}")

    grid = build_grid(true_lits, N)
    conflicts = find_conflicts(grid, D)

    print_grid(grid, conflicts, D)
    print_summary(grid, conflicts, D)

    # Exit with non-zero status if there are conflicts
    sys.exit(1 if conflicts else 0)


if __name__ == "__main__":
    main()
