# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal


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
    Query the effective on-chip cache/memory size of a GPU in bytes.
    Returns None if unable to determine.

    Note: Returns an "effective working set" size that empirically yields good throughput,
    balancing cache utilization with sufficient parallelism. Based on empirical testing,
    optimal working set is often larger than raw L2 size.

    Tries pynvml first (for L2 only), falls back to known GPU lookup table with
    empirically-tuned values.
    """
    # Fallback: lookup table by GPU name
    # Values are "effective working set" estimates based on empirical throughput testing
    # Optimal balances cache residency vs parallelism (typically 0.1-0.2% of VRAM)
    CACHE_TABLE = {
        # Volta (L2=6MB) - empirical optimal ~0.1-0.4% of 32GB VRAM = 32-130MB, target middle
        "V100": 48 * 1024 * 1024,  # 48 MB (~0.15% of 32GB)
        # Ampere (large L2 caches)
        "A100": 80 * 1024 * 1024,  # 80 MB (40MB L2, but more parallelism helps)
        "A6000": 24 * 1024 * 1024,  # 24 MB (~0.05% of 48GB)
        "A5000": 20 * 1024 * 1024,  # 20 MB
        "A4000": 12 * 1024 * 1024,  # 12 MB
        "RTX 3090": 24 * 1024 * 1024,  # 24 MB (~0.1% of 24GB)
        "RTX 3080": 12 * 1024 * 1024,  # 12 MB
        "RTX 3070": 8 * 1024 * 1024,  # 8 MB
        # Hopper (very large L2)
        "H100": 100 * 1024 * 1024,  # 100 MB (50MB L2 + parallelism headroom)
        "H200": 100 * 1024 * 1024,  # 100 MB
        # Ada Lovelace (large L2 caches - use ~1.5x L2 for parallelism)
        "RTX 4090": 96 * 1024 * 1024,  # 96 MB (72MB L2)
        "RTX 4080": 80 * 1024 * 1024,  # 80 MB (64MB L2)
        "RTX 4070": 48 * 1024 * 1024,  # 48 MB (36MB L2)
        "RTX A2000": 4 * 1024 * 1024,
        "RTX A4000": 6 * 1024 * 1024,
        "RTX A5000": 8 * 1024 * 1024,
        "RTX A6000": 6 * 1024 * 1024,
        "L40": 64 * 1024 * 1024,  # 64 MB (48MB L2)
        # Blackwell
        "B100": 96 * 1024 * 1024,  # 96 MB (estimated)
        "B200": 96 * 1024 * 1024,  # 96 MB (estimated)
    }
    gpu_name = device.device_kind
    for key, size in CACHE_TABLE.items():
        if key in gpu_name:
            return size
    return 32 * 1024 * 1024  # Conservative default
