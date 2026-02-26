#!/usr/bin/env python3
"""
Visualize a list of signed numbers (1 to n²) as an n×n grid.
Numbers are interpreted column-wise: indices 0 to n-1 are column 1, etc.
Positive and negative numbers are displayed differently.
"""

def draw_grid(numbers):
    """
    Draw an n×n grid from a list of signed numbers.
    
    Args:
        numbers: List of integers from 1 to n² (positive or negative)
                 Interpreted column-wise.
    """
    n_squared = len(numbers)
    n = int(n_squared ** 0.5)
    
    if n * n != n_squared:
        raise ValueError(f"List length {n_squared} is not a perfect square")
    
    # Build the grid (column-wise interpretation)
    # numbers[0] to numbers[n-1] = column 0
    # numbers[n] to numbers[2n-1] = column 1, etc.
    grid: list[list[int]] = [[0 for _ in range(n)] for _ in range(n)]
    
    for idx, num in enumerate(numbers):
        col = idx // n
        row = idx % n
        grid[row][col] = num
    
    # Print the grid
    print(f"\n{n}×{n} Grid (+ = positive, - = negative):\n")
    
    # Header
    print("    " + " ".join(f"{c:>4}" for c in [f"C{str(x):<2}" for x in range(n)]))
    print("    " + "-" * (5 * n))
    
    for row_idx, row in enumerate(grid):
        row_str = f"R{row_idx:<2}|"
        for num in row:
            if num > 0:
                row_str += " [+] "  # Positive
            elif num < 0:
                row_str += " [-] "  # Negative
            else:
                pass
        print(row_str)
    
    print()
    
    # Also print with actual numbers
    print("With values:")
    print("    " + " ".join(f"{c:>5}" for c in ["C"+str(x) for x in range(n)]))
    print("    " + "-" * (6 * n))
    
    for row_idx, row in enumerate(grid):
        row_str = f"R{row_idx:<2}|"
        for num in row:
            row_str += f"{num:>5} "
        print(row_str)


def draw_grid_visual(numbers: list[int], pos_char='█', neg_char='░'):
    """
    Draw a more visual representation of the grid.
    
    Args:
        numbers: List of integers from 1 to n² (positive or negative)
        pos_char: Character to use for positive numbers
        neg_char: Character to use for negative numbers
    """
    n_squared = len(numbers)
    n = int(n_squared ** 0.5)
    
    if n * n != n_squared:
        raise ValueError(f"List length {n_squared} is not a perfect square")
    
    # Build the grid (column-wise interpretation)
    grid: list[list[int]] = [[0 for _ in range(n)] for _ in range(n)]
    
    for idx, num in enumerate(numbers):
        col = idx // n
        row = idx % n
        grid[row][col] = num
    
    print(f"\n{n}×{n} Visual Grid:")
    print(f"  {pos_char} = positive, {neg_char} = negative\n")
    
    # Top border
    print("┌─" + "──" * n + "┐")
    
    for row in grid:
        line = "│ "
        for num in row:
            if num > 0:
                line += pos_char + " "
            elif num < 0:
                line += neg_char + " "
            else:
                pass # == 0, shouldn't be possible
        line = line.rstrip() + " │"
        print(line)
    
    # Bottom border
    print("└─" + "──" * n + "┘")


if __name__ == "__main__":
    # import argparse
    # import sys
    
    # parser = argparse.ArgumentParser(description="Visualize signed numbers as a grid")
    # parser.add_argument("numbers", nargs="*", type=int, 
    #                     help="List of signed integers (length must be a perfect square)")
    # parser.add_argument("-f", "--file", type=str,
    #                     help="Read numbers from file (one per line or space-separated)")
    # parser.add_argument("--pos", default="█", help="Character for positive (default: █)")
    # parser.add_argument("--neg", default="░", help="Character for negative (default: ░)")
    
    # args = parser.parse_args()
    
    # numbers = []
    
    # if args.file:
    #     with open(args.file, 'r') as f:
    #         content = f.read()
    #         numbers = [int(x) for x in content.split()]
    # elif args.numbers:
    #     numbers = args.numbers
    # else:
    #     # Demo with a 4x4 grid
    #     print("No input provided. Running demo with 4×4 grid...")
    #     numbers = [1, -2, 3, -4,    # Column 0
    #                -5, 6, -7, 8,    # Column 1
    #                9, -10, 11, -12, # Column 2
    #                -13, 14, -15, 16] # Column 3
    
    # if not numbers:
    #     print("Error: No numbers provided", file=sys.stderr)
    #     sys.exit(1)
    
    # draw_grid(numbers)
    # draw_grid_visual(numbers, args.pos, args.neg)

    numbers = "-1 -2 -3 4 -5 -6 -7 -8 -9 -10 11 -12 -13 -14 -15 -16 -17 -18 -19 -20 -21 -22 -23 -24 25 -26 -27 -28 -29 -30 -31 -32 -33 -34 35 -36 -37 -38 39 -40 -41 -42 -43 -44 -45 -46 -47 -48 -49 50 -51 -52 -53 -54 -55 -56 -57 -58 -59 -60 -61 -62 63 -64 -65 -66 -67 -68 69 -70 -71 -72 73 -74 -75 -76 -77 -78 -79 -80 -81"
    numbers = [int(n) for n in numbers.split()]
    draw_grid(numbers)
    draw_grid_visual(numbers, "█", "░")

