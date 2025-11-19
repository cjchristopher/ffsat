import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal
from typing import Optional as Opt

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


