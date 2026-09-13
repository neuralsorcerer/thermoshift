# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shard layout, expected row counts and committed-file verification."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from thermoshift.config import SPLITS, Config, boolean
from thermoshift.filesystem import dataset_path, read_json, sha256
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
    try:
        path = dataset_path(root, state_path(".", shard_id))
        if not path.exists():
            return False
        doc = read_json(path)
        if (
            doc["fingerprint"] != config.fingerprint
            or type(doc["shard_id"]) is not int
            or doc["shard_id"] != shard_id
            or doc["provenance_sha256"] != sha256(dataset_path(root, "run_config.json"))
        ):
            return False
        expected = expected_shard(config, shard_id)
        if not isinstance(doc["files"], list) or len(doc["files"]) != len(expected):
            return False
        if {f["path"]: f["rows"] for f in doc["files"]} != expected:
            return False
        for f in doc["files"]:
            full = dataset_path(root, f["path"])
            if (
                f["kind"] != f["path"].split("/")[1]
                or f["split"] != f["path"].split("/")[2]
                or type(f["shard_id"]) is not int
                or f["shard_id"] != shard_id
                or type(f["rows"]) is not int
                or type(f["bytes"]) is not int
                or f["bytes"] <= 0
                or not isinstance(f["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) is None
            ):
                return False
            if not full.is_file() or full.is_symlink() or full.stat().st_size != f["bytes"]:
                return False
            if verify_hash and sha256(full) != f["sha256"]:
                return False
            # State files are resumable checkpoints rather than a trust
            # boundary.  Validate the inexpensive Parquet footer as well as
            # its recorded digest so a self-consistent but malformed
            # checkpoint is regenerated instead of later being finalized.
            with pq.ParquetFile(full) as parquet:
                if parquet.metadata.num_rows != f["rows"] or not parquet.schema_arrow.equals(
                    SCHEMAS[f["kind"]], check_metadata=True
                ):
                    return False
        return True
    except (OSError, KeyError, ValueError, TypeError):
        return False


def remove_temporaries(root: str | Path, shard_id: int) -> None:
    """Remove known temporary files after verifying the committed shard."""
    for kind in SCHEMAS:
        for split in SPLITS:
            relative = Path(shard_path(kind, split, shard_id)).with_suffix(".parquet.inprogress")
            dataset_path(root, relative).unlink(missing_ok=True)
