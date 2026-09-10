# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shard layout, expected row counts and committed-file verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thermoshift.config import SPLITS, Config, boolean
from thermoshift.filesystem import read_json, sha256
from thermoshift.random import split_codes
from thermoshift.schema import SCHEMAS


def shard_path(kind: str, split: str, shard_id: int) -> str:
    """Return the relative Parquet path for a shard partition."""
    return f"data/{kind}/{split}/{shard_id // 1000:05d}/part-{shard_id:08d}.parquet"


def expected_shard(config: Config, shard_id: int) -> dict[str, int]:
    """Compute exact nonempty partition row counts for a shard."""
    start, stop = config.bounds(shard_id)
    ids = np.arange(start, stop, dtype=np.int64)
    codes = split_codes(ids, config.seed)
    lengths = np.minimum(config.episode_steps, config.rows - ids * config.episode_steps)
    result = {}
    for i, split in enumerate(SPLITS):
        count = int(lengths[codes == i].sum())
        if count:
            for kind in SCHEMAS:
                result[shard_path(kind, split, shard_id)] = count
    return result


def state_path(root: str | Path, shard_id: int) -> Path:
    """Return the committed shard-state path."""
    return Path(root) / "_state" / f"shard-{shard_id:08d}.json"


def shard_complete(
    root: str | Path, config: Config, shard_id: int, verify_hash: bool = True
) -> bool:
    """Check committed shard metadata, file sizes and optional checksums."""
    boolean(verify_hash, "verify_hash")
    path = state_path(root, shard_id)
    if not path.exists():
        return False
    try:
        doc = read_json(path)
        if (
            doc["fingerprint"] != config.fingerprint
            or doc["shard_id"] != shard_id
            or doc["provenance_sha256"] != sha256(Path(root) / "run_config.json")
        ):
            return False
        expected = expected_shard(config, shard_id)
        if len(doc["files"]) != len(expected):
            return False
        if {f["path"]: f["rows"] for f in doc["files"]} != expected:
            return False
        for f in doc["files"]:
            full = Path(root) / f["path"]
            if (
                f["kind"] != f["path"].split("/")[1]
                or f["split"] != f["path"].split("/")[2]
                or f["shard_id"] != shard_id
                or type(f["bytes"]) is not int
                or f["bytes"] <= 0
            ):
                return False
            if not full.is_file() or full.is_symlink() or full.stat().st_size != f["bytes"]:
                return False
            if verify_hash and sha256(full) != f["sha256"]:
                return False
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def remove_temporaries(root: str | Path, shard_id: int) -> None:
    """Remove known temporary files after verifying the committed shard."""
    for kind in SCHEMAS:
        for split in SPLITS:
            (Path(root) / shard_path(kind, split, shard_id)).with_suffix(
                ".parquet.inprogress"
            ).unlink(missing_ok=True)
