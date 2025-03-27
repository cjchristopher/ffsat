import sys

## ONLY WORKS FOR CARD AND EO CONSTRAINTS CURRENTLY

def parse_cardinality_constraint(card, vars):
    # Convert cardinality constraint
    vars = [int(v) for v in vars]
    sign = "+"
    try:
        card = int(card)
    except Exception:
        strict,start = 0,2 if '=' in card else 1,1 # we need to adjust by 1 for '<' or '>'
        if card.startswith(">"):
            card = int(card[start:]) + strict #e.g. '>3' needs to get to '>=4'.
        elif card.startswith("<"):
            card = -int(card[start:]) + strict #e.g. '<3' needs to get to '>=-2'
            sign = "-"

    # Convert variables to PBO format (xN for positive, ~xN for negative)
    pbo_vars = [f"{sign}1 {'~x' if v<0 else 'x'}{abs(v)}" for v in vars]
    pbo_constraint = " ".join(pbo_vars) + f" >= {card};"

    return pbo_constraint


def parse_exactly_one_constraint(vars):
    vars = [int(v) for v in vars]

    pbo_vars = [f"+1 {'~x' if v<0 else 'x'}{abs(v)}" for v in vars]

    pbo_constraint = " ".join(pbo_vars) + " = 1;"
    return pbo_constraint


def convert_file(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            words = line.split(" ")
            line_type = words[0]
            match line_type:
                case "d":
                    pbo_constraint = parse_cardinality_constraint(words[1], words[2:-1])
                    outfile.write(pbo_constraint + "\n")
                case "eo":
                    pbo_constraint = parse_exactly_one_constraint(words[1:-1])
                    outfile.write(pbo_constraint + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_cardinality_pbo.py <input_file> <output_file>")
    else:
        convert_file(sys.argv[1], sys.argv[2])
