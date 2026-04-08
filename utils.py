# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

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

    # Cache configuration
    dft_cache: str = ""

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


def get_gpu_l2_cache_size(device) -> int | None:
    """
    Query total on-chip cache capacity of a GPU in bytes.
    Returns None if unable to determine.

    Returns a model-based estimate of L1 + L2 (+ L3 if present) cache capacity.
    Function name is kept for backward compatibility with existing call sites.
    """
    # Lookup table by GPU name.
    # Values target total cache budget (L1 + L2 + L3 where present), in bytes.
    # For NVIDIA parts listed here, totals are L1+L2 (no dedicated L3 on these models).
    CACHE_TABLE = {
        "V100": int(16 * 1024 * 1024),  # ~6MB L2 + ~10MB aggregate L1
        # Ampere
        "A100": int(60.25 * 1024 * 1024),  # 40MB L2 + 20.25MB aggregate L1
        "A6000": int(16.5 * 1024 * 1024),  # 6MB L2 + 10.5MB aggregate L1
        "A5000": int(14 * 1024 * 1024),  # 6MB L2 + 8MB aggregate L1
        "A4000": int(10 * 1024 * 1024),  # 4MB L2 + 6MB aggregate L1
        "RTX 3090": int(16.25 * 1024 * 1024),  # 6MB L2 + 10.25MB aggregate L1
        "RTX 3080": int(13.5 * 1024 * 1024),  # 5MB L2 + 8.5MB aggregate L1
        "RTX 3070": int(9.75 * 1024 * 1024),  # 4MB L2 + 5.75MB aggregate L1
        # Hopper
        "H100": int(80 * 1024 * 1024),  # 50MB L2 + ~30MB aggregate L1
        "H200": int(83 * 1024 * 1024),  # 50MB L2 + ~33MB aggregate L1
        # Ada Lovelace
        "RTX 4090": int(88 * 1024 * 1024),  # 72MB L2 + 16MB aggregate L1
        "RTX 4080": int(73.5 * 1024 * 1024),  # 64MB L2 + 9.5MB aggregate L1
        "RTX 4070": int(41.75 * 1024 * 1024),  # 36MB L2 + 5.75MB aggregate L1
        "RTX A2000": int(7.25 * 1024 * 1024),  # 4MB L2 + 3.25MB aggregate L1
        "RTX A4000": int(10 * 1024 * 1024),  # 4MB L2 + 6MB aggregate L1
        "RTX A5000": int(14 * 1024 * 1024),  # 6MB L2 + 8MB aggregate L1
        "RTX A6000": int(16.5 * 1024 * 1024),  # 6MB L2 + 10.5MB aggregate L1
        "L40": int(65.75 * 1024 * 1024),  # 48MB L2 + ~17.75MB aggregate L1
        # Blackwell
        "B100": int(128 * 1024 * 1024),  # Estimated total cache
        "B200": int(160 * 1024 * 1024),  # Estimated total cache
    }
    gpu_name = device.device_kind
    for key, size in CACHE_TABLE.items():
        if key in gpu_name:
            return size
    return 32 * 1024 * 1024  # Conservative default
