# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Verify a release and read ordered pairs under a stable dataset lease."""

from __future__ import annotations

import re
from collections.abc import Iterator
from itertools import zip_longest
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from thermoshift._checks import require
from thermoshift.config import SPLITS, Config, boolean, integer
from thermoshift.filesystem import dataset_path, read_json, sha256
from thermoshift.lifecycle import dataset_lock
from thermoshift.provenance import check_schema_documents, load_config, metadata_hashes
from thermoshift.schema import SCHEMAS
from thermoshift.shards import expected_shard, shard_path


def validated_manifest(root: str | Path, verify_hash: bool = True) -> tuple[Config, dict[str, Any]]:
    boolean(verify_hash, "verify_hash")
    with dataset_lock(root):
        return _validated_manifest(root, verify_hash)


def _validated_manifest(root, verify_hash=True):
    boolean(verify_hash, "verify_hash")
    root = Path(root)
    config = load_config(root)
    check_schema_documents(root)
    manifest = read_json(dataset_path(root, "manifest.json"))
    success = read_json(dataset_path(root, "_SUCCESS.json"))
    require(isinstance(manifest, dict), "manifest must be a JSON object")
    require(isinstance(success, dict), "completion marker must be a JSON object")
    require(
        success.get("metadata_sha256") == metadata_hashes(root),
        "release metadata checksum mismatch",
    )
    require(
        success.get("manifest_sha256") == sha256(root / "manifest.json"),
        "manifest is not finalized",
    )
    require(
        manifest.get("complete") is True and manifest.get("fingerprint") == config.fingerprint,
        "manifest/config mismatch",
    )
    files = manifest.get("files")
    require(isinstance(files, list), "manifest files must be a list")
    for item in files:
        require(isinstance(item, dict), "manifest file entries must be JSON objects")
        require(isinstance(item.get("path"), str), "manifest file path must be a string")
        require(isinstance(item.get("kind"), str) and item["kind"] in SCHEMAS, "file kind mismatch")
        require(item.get("split") in SPLITS, "file split mismatch")
        for name in ("rows", "bytes", "shard_id"):
            integer(item.get(name), f"file {name}", 0 if name == "shard_id" else 1)
        require(item["shard_id"] < config.shards, "shard range mismatch")
        require(
            isinstance(item.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None,
            "invalid file SHA-256",
        )
        require(
            item["path"] == shard_path(item["kind"], item["split"], item["shard_id"]),
            "file path/shard mismatch",
        )
    for name in ("rows", "buildings", "bytes"):
        integer(manifest.get(name), f"manifest {name}", 1)
    split_rows = manifest.get("split_rows")
    require(
        isinstance(split_rows, dict) and set(split_rows) == set(SPLITS),
        "manifest split rows must cover the configured splits",
    )
    for count in split_rows.values():
        integer(count, "split row count", 0)
    expected = {}
    for sid in range(config.shards):
        expected.update(expected_shard(config, sid))
    require(
        files == sorted(files, key=lambda f: f["path"]),
        "manifest files must have canonical path order",
    )
    require(
        not (root / "data").is_symlink()
        and not any(p.is_symlink() for p in (root / "data").rglob("*")),
        "symlinks are not allowed under data/",
    )
    require(len(files) == len(expected), "manifest file count mismatch")
    require(
        {f["path"]: f["rows"] for f in files} == expected,
        "manifest does not cover the configured rows",
    )
    actual = {p.relative_to(root).as_posix() for p in (root / "data").rglob("*.parquet")}
    require(actual == set(expected), "unexpected/missing Parquet file")
    require(manifest["buildings"] == config.buildings, "total building count mismatch")
    require(manifest["rows"] == config.rows, "total row count mismatch")
    require(
        manifest["bytes"] == sum(f["bytes"] for f in files), "manifest byte accounting mismatch"
    )
    for item in files:
        require(item["kind"] == item["path"].split("/")[1], "file kind mismatch")
        require(item["split"] == item["path"].split("/")[2], "file split mismatch")
        path = dataset_path(root, item["path"])
        require(path.stat().st_size == item["bytes"], f"file size mismatch: {item['path']}")
        if verify_hash:
            require(sha256(path) == item["sha256"], f"checksum mismatch: {item['path']}")
        with pq.ParquetFile(path) as parquet:
            require(parquet.metadata.num_rows == item["rows"], "Parquet row count mismatch")
            require(
                parquet.schema_arrow.equals(SCHEMAS[item["kind"]], check_metadata=True),
                "Arrow schema mismatch",
            )
    for split in SPLITS:
        count = sum(f["rows"] for f in files if f["kind"] == "logged" and f["split"] == split)
        require(manifest["split_rows"][split] == count, "split row count mismatch")
    return config, manifest


def iter_pairs(
    root: str | Path, split: str | None = None, batch_size: int = 65_536
) -> Iterator[tuple[dict[str, Any], pa.Table, pa.Table]]:
    """Yield a stable, ordered snapshot; keep the generator open during reading.

    Checks file checksums before yielding. Public callers must exhaust or close
    the iterator to release its exclusive lifecycle lease.
    """
    integer(batch_size, "batch_size", 1)
    require(split is None or split in SPLITS, "unknown split")
    with dataset_lock(root):
        _, manifest = _validated_manifest(root)
        yield from _iter_pairs(root, manifest, split, batch_size)


def ordered_files(
    manifest: dict, kind: str | None = None, split: str | None = None
) -> list[dict[str, Any]]:
    """Use numeric shard IDs: filename padding is a minimum, not a size limit."""
    return sorted(
        (
            item
            for item in manifest["files"]
            if (kind is None or item["kind"] == kind) and (split is None or item["split"] == split)
        ),
        key=lambda item: (item["kind"], item["split"], item["shard_id"]),
    )


def _iter_pairs(root, manifest, split=None, batch_size=65_536):
    root = Path(root)
    for item in ordered_files(manifest, kind="logged", split=split):
        oracle_path = item["path"].replace("data/logged/", "data/oracle/", 1)
        with (
            pq.ParquetFile(root / item["path"]) as logged,
            pq.ParquetFile(root / oracle_path) as oracle,
        ):
            left = logged.iter_batches(batch_size=batch_size)
            right = oracle.iter_batches(batch_size=batch_size)
            for x, o in zip_longest(left, right):
                require(x is not None and o is not None, "paired file lengths differ")
                require(x.num_rows == o.num_rows, "paired batch lengths differ")
                for name in ("row_id", "building_id", "step"):
                    require(x[name].equals(o[name]), "logged/oracle join mismatch")
                yield item, pa.Table.from_batches([x]), pa.Table.from_batches([o])
