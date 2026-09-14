# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Immutable scientific and storage configuration; scheduling is kept separate."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

SPLITS = ("train", "validation", "test", "test_heatwave", "test_sensor")
LEVELS = (0.0, 0.5, 1.0)


def integer(value: object, name: str, minimum: int = 0, maximum: int | None = None) -> int:
    """Validate Python integer API arguments; reject booleans and silent coercions."""
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(
            f"{name} must be an integer in [{minimum}, {maximum if maximum is not None else 'unbounded'}]"
        )
    return value


def boolean(value: object, name: str) -> bool:
    """Reject truthy strings and numbers at public API boundaries."""
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


@dataclass(frozen=True)
class Config:
    rows: int = 1_000_000
    seed: int = 42
    episode_steps: int = 168
    shard_buildings: int = 16_384
    exploration: float = 0.15
    hidden_confounding: bool = False
    comfort_weight: float = 0.30
    carbon_weight: float = 0.05
    compression: str = "zstd"
    compression_level: int = 3
    schema_version: str = "2.0"

    def __post_init__(self) -> None:
        for key in ("rows", "seed", "episode_steps", "shard_buildings", "compression_level"):
            if type(getattr(self, key)) is not int:
                raise ValueError(
                    f"{key} must be a Python integer, not {type(getattr(self, key)).__name__}"
                )
        if not 1 <= self.rows <= 2**63 - 1:
            raise ValueError("rows must fit a positive signed 64-bit integer")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if not 2 <= self.episode_steps <= 8760:
            raise ValueError("episode_steps must be in [2, 8760]")
        if not 1 <= self.shard_buildings <= 1_000_000:
            raise ValueError("shard_buildings must be in [1, 1000000]")
        for key in ("exploration", "comfort_weight", "carbon_weight"):
            value = getattr(self, key)
            # Booleans, strings and NumPy scalars are all rejected here: only
            # exact Python numbers fingerprint and serialize reproducibly.
            if type(value) not in (int, float):
                raise ValueError(
                    f"{key} must be a finite Python int or float, not {type(value).__name__}"
                )
            try:
                value = float(value)
            except OverflowError as error:
                raise ValueError(f"{key} must be a finite real number") from error
            if not math.isfinite(value):
                raise ValueError(f"{key} must be a finite real number")
            object.__setattr__(self, key, value)
        if not 0 < self.exploration <= 1:
            raise ValueError("exploration must be in (0, 1]")
        if not (0 <= self.comfort_weight < 1e6 and 0 <= self.carbon_weight < 1e6):
            raise ValueError("reward weights must be finite and in [0, 1000000)")
        if type(self.hidden_confounding) is not bool:
            raise ValueError("hidden_confounding must be boolean")
        if (
            type(self.compression) is not str
            or self.compression != "zstd"
            or not 1 <= self.compression_level <= 19
        ):
            raise ValueError("this version supports zstd compression levels 1 through 19")
        if type(self.schema_version) is not str or self.schema_version != "2.0":
            raise ValueError("unsupported schema version")

    @property
    def buildings(self) -> int:
        return (self.rows + self.episode_steps - 1) // self.episode_steps

    @property
    def shards(self) -> int:
        return (self.buildings + self.shard_buildings - 1) // self.shard_buildings

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def bounds(self, shard_id: int) -> tuple[int, int]:
        integer(shard_id, "shard_id", 0, self.shards - 1)
        start = shard_id * self.shard_buildings
        return start, min(start + self.shard_buildings, self.buildings)
