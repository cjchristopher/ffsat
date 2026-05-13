# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
import sys
import math

def parse_cardinality_constraint(card, vars) -> str:
    # Convert cardinality constraint
    sign = "+"
    try:
        card = int(card)
    except Exception:
        strict, start = 0, 2 if "=" in card else 1  # we need to adjust by 1 for '<' or '>'
        if card.startswith(">"):
            card = int(card[start:]) + strict  # e.g. '>3' needs to get to '>=4'.
        elif card.startswith("<"):
            card = -int(card[start:]) + strict  # e.g. '<3' needs to get to '>=-2'
            sign = "-"

    # Convert variables to PBO format (xN for positive, ~xN for negative)
    pbo_vars = [f"{sign}1 {'~x' if v < 0 else 'x'}{abs(v)}" for v in vars]
    pbo_constraint = " ".join(pbo_vars) + f" >= {card};"
    return pbo_constraint


def parse_exactly_k_constraint(card, vars) -> str:
    # Convert cardinality constraint
    sign = "+"
    try:
        card = int(card)
    except Exception as e:
        raise e

    # Convert variables to PBO format (xN for positive, ~xN for negative)
    pbo_vars = [f"{sign}1 {'~x' if v < 0 else 'x'}{abs(v)}" for v in vars]
    pbo_constraint = " ".join(pbo_vars) + f" = {card};"
    return pbo_constraint


def parse_exactly_one_constraint(vars) -> str:
    pbo_vars = [f"+1 {'~x' if v < 0 else 'x'}{abs(v)}" for v in vars]
    pbo_constraint = " ".join(pbo_vars) + " = 1;"
    return pbo_constraint


def parse_at_most_one_constraint(vars) -> str:
    # ideally "+1 xY +1 ~xZ +1 xA <= 1", but sat4j only accepts >= or =.
    # so flip the signs - e.g. {-1 [~]xX} >= -1
    pbo_vars = [f"-1 {'~x' if v < 0 else 'x'}{abs(v)}" for v in vars]
    pbo_constraint = " ".join(pbo_vars) + " >= -1;"
    return pbo_constraint


def parse_cnf_constraint(vars) -> str:
    pbo_vars = [f"+1 {'~x' if v < 0 else 'x'}{abs(v)}" for v in vars]
    op = ">=" if len(vars) > 1 else "="
    pbo_constraint = " ".join(pbo_vars) + f" {op} 1;"
    return pbo_constraint


def parse_not_all_equal_constraint(vars) -> str:
    # NAE implies that both the CNF and ~CNF read of the clause is SAT.
    pbo_constraint = parse_cnf_constraint(vars)
    pbo_constraint += parse_cnf_constraint([-1 * v for v in vars])
    return pbo_constraint


def parse_xor_constraint(vars, var_cnt) -> tuple[str, int]:
    # Encode odd parity as sum(literals) - 2*k = 1, where k is an integer
    aux_cnt = math.floor(math.log(len(vars), 2)) + 1
    pbo_vars = [f"+1 {'~x' if v < 0 else 'x'}{abs(v)}" for v in vars]
    aux_vars = [f"-{2**aux} x{var_cnt+aux}" for aux in range(1, aux_cnt)]
    pbo_constraint = " ".join(pbo_vars+aux_vars) + " = 1;"
    return pbo_constraint, len(aux_vars)


def scan_true_counts(input_file, header_var_cnt) -> tuple[int, int, int, int]:
    source_var_cnt = 0
    source_cls_cnt = 0
    converted_var_cnt = 0
    converted_cls_cnt = 0
    next_aux_var_base = header_var_cnt

    with open(input_file, "r") as infile:
        for line in infile:
            words = line.split()
            if not words:
                continue
            if words[0] in ("c", "p"):
                continue

            if words[0] == "h":
                words = words[1:]

            line_type = words[0]

            if line_type in ("d", "card", "k", "ek"):
                vars = [int(v) for v in words[2:-1]]
            elif line_type in ("e", "eo", "a", "amo", "x", "xor", "n", "nae", "cnf"):
                vars = [int(v) for v in words[1:-1]]
            else:
                # Untyped lines are treated as plain CNF.
                vars = [int(v) for v in words[0:-1]]

            if vars:
                max_var = max(abs(v) for v in vars)
                source_var_cnt = max(source_var_cnt, max_var)
                converted_var_cnt = max(converted_var_cnt, max_var)

            if line_type in ("x", "xor"):
                _, aux_cnt = parse_xor_constraint(vars, next_aux_var_base)
                next_aux_var_base += aux_cnt
                converted_var_cnt = max(converted_var_cnt, next_aux_var_base)

            source_cls_cnt += 1
            converted_cls_cnt += 1

    return source_var_cnt, source_cls_cnt, converted_var_cnt, converted_cls_cnt


def convert_file(input_file, output_file) -> None:
    with open(input_file, "r") as infile:
        header_var_cnt = 0
        header_cls_cnt = 0
        for line in infile:
            words = line.split()
            if not words:
                continue
            if words[0] == "c":
                continue
            if not header_var_cnt:
                if words[0] == "p" and words[1] in ("opb", "pbo"):
                    raise ValueError(
                        "Input appears to already be OPB/PBO format. "
                        "hybrid_to_pbo.py only converts non-OPB formats (e.g. cnf/hybrid)."
                    )
                if words[0] == "p" and words[1] in ("hybrid", "cnf"):
                    header_var_cnt = int(words[2])
                    header_cls_cnt = int(words[3])
                    continue

        if not header_var_cnt:
            raise ValueError(
                "Unsupported or missing input header. Expected 'p cnf ...' or 'p hybrid ...'."
            )

    source_var_cnt, source_cls_cnt, converted_var_cnt, converted_cls_cnt = scan_true_counts(
        input_file, header_var_cnt
    )

    if source_var_cnt != header_var_cnt:
        print(
            f"Warning: header variable count ({header_var_cnt}) does not match "
            f"actual source variable count ({source_var_cnt}).",
            file=sys.stderr,
        )
    if source_cls_cnt != header_cls_cnt:
        print(
            f"Warning: header constraint count ({header_cls_cnt}) does not match "
            f"actual source constraint count ({source_cls_cnt}).",
            file=sys.stderr,
        )

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        pbo_cons = []
        next_aux_var_base = header_var_cnt
        for line in infile:
            words = line.split()
            if not words:
                continue
            if words[0] == "c":
                continue
            if words[0] == "p":
                continue

            if words[0] == "h":
                words = words[1:]

            line_type = words[0]
            vars = [int(v) for v in words[1:-1]]

            match line_type:
                case "d" | "card":
                    pbo_constraint = parse_cardinality_constraint(vars[0], vars[1:])
                case "k" | "ek":
                    pbo_constraint = parse_exactly_k_constraint(vars[0], vars[1:])
                case "e" | "eo":
                    pbo_constraint = parse_exactly_one_constraint(vars)
                case "a" | "amo":
                    pbo_constraint = parse_at_most_one_constraint(vars)
                case "x" | "xor":
                    pbo_constraint, aux_cnt = parse_xor_constraint(vars, next_aux_var_base)
                    next_aux_var_base += aux_cnt
                case "n" | "nae":
                    pbo_constraint = parse_not_all_equal_constraint(vars)
                case "cnf" | _:  # CNF
                    if line_type != "cnf":
                        vars = [int(v) for v in words[0:-1]]
                    pbo_constraint = parse_cnf_constraint(vars)
            pbo_cons.append(pbo_constraint)

        # Write file.
        outfile.write(f"* #variable= {converted_var_cnt} #constraint= {converted_cls_cnt} *\n")
        for con in pbo_cons:
            outfile.write(con + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hybrid_to_pbo.py <input_file> <output_file>")
    else:
        convert_file(sys.argv[1], sys.argv[2])
