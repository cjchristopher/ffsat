import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple
from typing import Optional as Opt

import typer

from boolean_whf import Clause, ClauseGroup, Objective, class_idno, empty_validator
from sat_loader import Formula


class Validators(NamedTuple):
    xor: Objective
    cnf: Objective
    eo: Objective
    nae: Objective
    card: Objective
    amo: Objective


def valid_choice(value: str) -> int:
    choice_map = {"1": "full", "2": "types", "3": "lens"}
    if int(value) not in {1, 2, 3}:
        raise typer.BadParameter("Choice must be 1, 2, or 3")
    return choice_map[value]


def preprocess_to_matrix(sat: Formula, mode: Opt[int], threshold: int = 0) -> tuple[tuple[Objective, ...], Validators]:
    clause_grps: dict[str, ClauseGroup] = {}
    if mode is not None and mode > 0:
        choice = valid_choice(str(mode))
    else:
        print("Please see the following clause type and clause length breakdowns and select an option:")
        print(
            "Types count:\n\t",
            [f"{x}: {len(y)} (length counts - {sat.stats[x]})" for (x, y) in sat.clauses_type.items() if y],
        )
        print("Lengths counts\n\t", [f"len {x}: {len(y)} clauses" for (x, y) in sat.clauses_len.items()])
        print(
            "\tOptions:\n"
            + "\t\t1: Full combine. Use single monolithlic array with all clauses appropriately padded\n"
            + "\t\t2: By type. Separate padded array for each clause type\n"
            + "\t\t3: By length. Separate (possibly minor padding) for each clause length (or length cluster)"
        )
        choice = typer.prompt("Options", type=valid_choice, default="3")

    n_var = sat.n_var
    # We need the breakdown by clause type anyway for quick validation.
    # TODO: No longer required - left for backwards compatibility with the unsharded code.
    # Once unsharded is factored, this first loop should become the last block in the match/case below.
    for c_type, c_list in sat.clauses_type.items():
        if c_list:
            # Clauses present, do FFTs if user partition choice was by type.
            clause_grps[c_type] = ClauseGroup(c_list, n_var, do_fft=(choice == "types"), clause_type=c_type)
        else:
            clause_grps[c_type] = None

    match choice:
        case "full":
            clause_grps["full"] = ClauseGroup(sat.clauses_all, n_var, do_fft=True, clause_type="all")
        case "lens":
            # thresh = 0
            # TODO: Implement clustering with some threshold?
            for c_len, c_list in sat.clauses_len.items():
                clause_grps[c_len] = ClauseGroup(c_list, n_var, do_fft=True, clause_type="mixed")
        case _:
            # type already proesssed above.
            pass

    # Process groups (in parallel).
    def process_groups(clause_grps: dict[str, ClauseGroup], max_workers: Opt[int] = None):
        def process(grp: ClauseGroup):
            aux_grp = grp.process()
            return aux_grp

        with ThreadPoolExecutor(max_workers=max_workers) as tpool:
            tasks = [tpool.submit(process, grp) for grp in clause_grps.values() if grp]
            res = []
            for task in tasks:
                res.append(task.result())

        # Collect detected unit literals
        lits = set()
        for r in res:
            if r is not None:
                lits.update(r)
        if lits:
            unit_lits = [Clause("cnf", [lit], 0) for lit in lits]
            unit_lits = ClauseGroup(unit_lits, len(unit_lits), do_fft=True, clause_type="cnf")
            unit_lits.process()
            return [unit_lits]
        return []

    objectives = process_groups(clause_grps, max_workers=min(len(clause_grps), 8))
    empty_Validation = empty_validator(n_var)
    validation = {}

    # TODO: As above, separate validation no longer required, but left for compat with unsharded.
    for grp_type, grp in clause_grps.items():
        if not grp:
            # No clause set, so in clause_grps for validation, add the empty.
            validation[grp_type] = empty_Validation
            continue

        # Valid clause set
        objective = grp.get()
        if grp_type in class_idno:
            # Required for validation
            validation[grp_type] = objective
        if choice == "types" or (grp_type not in class_idno):
            # Optimisation objective
            objectives.append(objective)

    objectives = tuple(sorted(objectives, key=lambda x: x.clauses.lits.shape[-1]))
    validation = Validators(**validation)
    return objectives, validation


# TODO: Implement config file passing and setting instead of all command line switches.
@dataclass
class FFSATConfig:
    """Configuration for FFSAT solver."""

    # Solver configuration
    solver_type: Literal["pgd", "hj_prox"] = "pgd"
    solver_params: dict[str, Any] = field(
        default_factory=lambda: {
            "maxiter": 50000,
            "projection": "box",
            "projection_params": (-1, 1),
        }
    )

    # Formula partitioning configuration
    partition_strategy: Literal["full", "types", "lens"] = "types"

    # Cache configuration
    cache_file: Opt[str] = None

    # Run configuration
    tasks: int = 32
    batch_size: int = 16
    max_time: float = 300.0  # Max run time in seconds
    timeout_behavior: Literal["early_stop", "continue_until_batch"] = "continue_until_batch"

    # Output configuration
    verbose: bool = True

    @classmethod
    def from_file(cls, filepath: str) -> "FFSATConfig":
        """Load configuration from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath, "r") as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def to_file(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.__dict__, f, indent=2)
