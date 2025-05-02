import sys
import math

def parse_cardinality_constraint(card, vars):
    # Convert cardinality constraint
    sign = "+"
    try:
        card = int(card)
    except Exception:
        strict, start = 0, 2 if "=" in card else 1, 1  # we need to adjust by 1 for '<' or '>'
        if card.startswith(">"):
            card = int(card[start:]) + strict  # e.g. '>3' needs to get to '>=4'.
        elif card.startswith("<"):
            card = -int(card[start:]) + strict  # e.g. '<3' needs to get to '>=-2'
            sign = "-"

    # Convert variables to PBO format (xN for positive, ~xN for negative)
    pbo_vars = [f"{sign}1 {'~x' if v < 0 else 'x'}{abs(v)}" for v in vars]
    pbo_constraint = " ".join(pbo_vars) + f" >= {card};"
    return pbo_constraint


def parse_exactly_one_constraint(vars):
    pbo_vars = [f"+1 {'~x' if v < 0 else 'x'}{abs(v)}" for v in vars]
    pbo_constraint = " ".join(pbo_vars) + " = 1;"
    return pbo_constraint


def parse_at_most_one_constraint(vars):
    # ideally "+1 xY +1 ~xZ +1 xA <= 1", but sat4j only accepts >= or =.
    # so flip the signs - e.g. {-1 [~]xX} >= -1
    pbo_vars = [f"-1 {'~x' if v < 0 else 'x'}{abs(v)}" for v in vars]
    pbo_constraint = " ".join(pbo_vars) + " >= -1;"
    return pbo_constraint


def parse_cnf_constraint(vars):
    pbo_vars = [f"+1 {'~x' if v < 0 else 'x'}{abs(v)}" for v in vars]
    pbo_constraint = " ".join(pbo_vars) + " >= 1;"
    return pbo_constraint


def parse_not_all_equal_constraint(vars):
    # NAE implies that both the CNF and ~CNF read of the clause is SAT.
    pbo_constraint = parse_cnf_constraint(vars)
    pbo_constraint += parse_cnf_constraint([-1 * v for v in vars])
    return pbo_constraint


def parse_xor_constraint(vars, var_cnt):
    # Introduces free auxiliary variables so needs to know the total variable count.
    aux_cnt = math.floor(math.log(len(vars), 2)) + 1
    pbo_vars = [f"+1 {'~x' if v < 0 else 'x'}{abs(v)}" for v in vars]
    aux_vars = [f"+{2**aux} x{var_cnt+aux}" for aux in range(1, aux_cnt)]
    pbo_constraint = " ".join(pbo_vars+aux_vars) + " = 1;"
    return pbo_constraint, aux_cnt


def convert_file(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        var_cnt = 0
        pbo_cons = []
        for line in infile:
            words = line.split(" ")
            if not var_cnt:
                if words[0] == "p" and words[1] in ("hybrid", "pbo"):
                    var_cnt = int(words[2])
                    cls_cnt = int(words[3])
                    continue

            line_type = words[0]
            vars = [int(v) for v in words[1:-1]]
            match line_type:
                case "d" | "card":
                    pbo_constraint = parse_cardinality_constraint(vars[0], vars[1:])
                case "e" | "eo":
                    pbo_constraint = parse_exactly_one_constraint(vars)
                case "a" | "amo":
                    pbo_constraint = parse_at_most_one_constraint(vars)
                case "x" | "xor":
                    pbo_constraint, aux_cnt = parse_xor_constraint(vars, var_cnt)
                    var_cnt += aux_cnt
                case "n" | "nae":
                    pbo_constraint = parse_not_all_equal_constraint(vars)
                case "cnf" | _:  # CNF
                    if line_type != "cnf":
                        vars = [int(v) for v in words[0:-1]]
                    pbo_constraint = parse_cnf_constraint(vars)
            pbo_cons.append(pbo_constraint)

        # Write file.
        outfile.write(f"* #variable= {var_cnt} #constraint= {cls_cnt}\n*")
        for con in pbo_cons:
            outfile.write(con + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hybrid_to_pbo.py <input_file> <output_file>")
    else:
        convert_file(sys.argv[1], sys.argv[2])
