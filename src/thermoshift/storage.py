# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Initialize, generate and finalize datasets with bounded worker scheduling."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from thermoshift.cards import write_card
from thermoshift.config import SPLITS, Config, integer
from thermoshift.filesystem import atomic_json, fsync_directory, read_json, sha256
from thermoshift.lifecycle import dataset_lock
from thermoshift.provenance import (
    check_schema_documents,
    load_config,
    metadata_hashes,
    runtime,
    source_digest,
)
from thermoshift.schema import SCHEMAS
from thermoshift.shards import (
    expected_shard,
    remove_temporaries,
    shard_complete,
    shard_path,
    state_path,
)
from thermoshift.simulator import simulate


def initialize(root: str | Path, config: Config) -> Config:
    """Initialize an empty directory or reopen a matching generation plan."""
    if not isinstance(config, Config):
        raise TypeError("config must be a Config instance")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with dataset_lock(root):
        path = root / "run_config.json"
        if path.exists():
            current = load_config(root, check_runtime=True)
            if current != config:
                raise ValueError(
                    "output contains a different configuration; use a new output directory"
                )
            check_schema_documents(root, repair_missing=True)
            return config
        # A process may have died before its first atomic config rename.
        for temporary in root.glob("run_config.json.tmp-*"):
            temporary.unlink()
        if any(p.name != ".lifecycle.lock" for p in root.iterdir()):
            raise ValueError("new dataset output directory must be empty")
        atomic_json(
            path,
            {
                "config": asdict(config),
                "fingerprint": config.fingerprint,
                "source_sha256": source_digest(),
                "runtime": runtime(),
            },
        )
        check_schema_documents(root, repair_missing=True)
    return config


def _generate_shard(root, config_dict, shard_id, batch_buildings):
    # Children keep their own shared lease if their parent is terminated.
    with dataset_lock(root, generating=True):
        return _generate_shard_locked(root, config_dict, shard_id, batch_buildings)


def _worker_initialize():
    """Limit Arrow pools only inside processes owned by this generator."""
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def _generate_shard_locked(root, config_dict, shard_id, batch_buildings):
    config = Config(**config_dict)
    root = Path(root)
    state = root / "_state"
    state.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with FileLock(str(state / f"shard-{shard_id:08d}.lock"), timeout=0):
        if shard_complete(root, config, shard_id):
            remove_temporaries(root, shard_id)
            return {
                "shard_id": shard_id,
                "status": "resumed",
                "seconds": time.perf_counter() - started,
            }
        writers, totals, temp_paths = {}, {}, {}
        try:
            # ExitStack attempts every close even if writing or a footer close
            # fails. Own the underlying handles separately so a failed writer
            # cannot keep the temporary open (particularly on Windows).
            with ExitStack() as resources:
                start, stop = config.bounds(shard_id)
                for low in range(start, stop, batch_buildings):
                    logged, oracle, codes = simulate(config, low, min(low + batch_buildings, stop))
                    for split_id, split in enumerate(SPLITS):
                        mask = pa.array(codes == split_id)
                        if not np.any(codes == split_id):
                            continue
                        for kind, table in (("logged", logged), ("oracle", oracle)):
                            relative = shard_path(kind, split, shard_id)
                            if relative not in writers:
                                final = root / relative
                                final.parent.mkdir(parents=True, exist_ok=True)
                                temporary = final.with_suffix(".parquet.inprogress")
                                temp_paths[relative] = temporary
                                handle = resources.enter_context(temporary.open("wb"))
                                writer = pq.ParquetWriter(
                                    handle,
                                    SCHEMAS[kind],
                                    compression=config.compression,
                                    compression_level=config.compression_level,
                                    use_dictionary=True,
                                    write_statistics=True,
                                    version="2.6",
                                )
                                resources.callback(writer.close)
                                writers[relative] = writer
                                totals[relative] = 0
                            part = table.filter(mask)
                            writers[relative].write_table(part, row_group_size=131_072)
                            totals[relative] += part.num_rows
            if totals != expected_shard(config, shard_id):
                raise RuntimeError("generated shard row accounting is inconsistent")
            files = []
            for relative, temporary in sorted(temp_paths.items()):
                with temporary.open("r+b") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, root / relative)
                final = root / relative
                fsync_directory(final.parent)
                files.append(
                    {
                        "path": relative,
                        "rows": totals[relative],
                        "bytes": final.stat().st_size,
                        "sha256": sha256(final),
                        "kind": relative.split("/")[1],
                        "split": relative.split("/")[2],
                        "shard_id": shard_id,
                    }
                )
            atomic_json(
                state_path(root, shard_id),
                {
                    "fingerprint": config.fingerprint,
                    "shard_id": shard_id,
                    "provenance_sha256": sha256(root / "run_config.json"),
                    "files": files,
                },
            )
        finally:
            for path in temp_paths.values():
                path.unlink(missing_ok=True)
    return {"shard_id": shard_id, "status": "written", "seconds": time.perf_counter() - started}


def generate(
    root: str | Path,
    workers: int = 1,
    batch_buildings: int = 512,
    rank: int = 0,
    world_size: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, int | float]:
    """Generate or resume assigned shards; single-rank runs also finalize.

    The optional progress callback receives one JSON string per completed shard."""
    root = Path(root).resolve()
    integer(workers, "workers", 1)
    integer(batch_buildings, "batch_buildings", 1)
    integer(world_size, "world_size", 1)
    integer(rank, "rank", 0, world_size - 1)
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    with dataset_lock(root, generating=True):
        config = load_config(root, check_runtime=True)
        check_schema_documents(root)
        (root / "_SUCCESS.json").unlink(missing_ok=True)
        (root / "validation.json").unlink(missing_ok=True)
        summary = _generate_locked(
            root, config, workers, batch_buildings, rank, world_size, progress
        )
    if world_size == 1:
        finalize(root)
    return summary


def _generate_locked(root, config, workers, batch_buildings, rank, world_size, progress):
    assigned_shards = range(rank, config.shards, world_size)
    shard_ids = iter(assigned_shards)
    summary = {"written": 0, "resumed": 0}
    started = time.perf_counter()
    workers = min(workers, len(assigned_shards))
    if workers == 0:
        return {**summary, "seconds": time.perf_counter() - started}

    def record(result):
        summary[result["status"]] += 1
        if progress is not None:
            progress(json.dumps(result))

    if workers == 1:
        for sid in shard_ids:
            record(_generate_shard(str(root), asdict(config), sid, batch_buildings))
    else:
        # At most 2*workers tasks queued; no billion-element future or row list.
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_initialize,
        ) as pool:
            pending = set()

            def submit_next():
                sid = next(shard_ids, None)
                if sid is not None:
                    pending.add(
                        pool.submit(
                            _generate_shard, str(root), asdict(config), sid, batch_buildings
                        )
                    )
                    return True
                return False

            for _ in range(2 * workers):
                if not submit_next():
                    break
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    record(future.result())
                    submit_next()
    summary["seconds"] = time.perf_counter() - started
    return summary


def finalize(root: str | Path) -> dict[str, Any]:
    """Verify all committed shards and write the manifest, card and completion marker."""
    root = Path(root)
    with dataset_lock(root):
        config = load_config(root, check_runtime=True)
        check_schema_documents(root)
        files = []
        for sid in range(config.shards):
            if not shard_complete(root, config, sid, verify_hash=True):
                raise ValueError(f"shard {sid} is incomplete; generate all ranks before finalizing")
            remove_temporaries(root, sid)
            files.extend(read_json(state_path(root, sid))["files"])
        expected = {f["path"] for f in files}
        actual = {p.relative_to(root).as_posix() for p in (root / "data").rglob("*.parquet")}
        if actual != expected:
            raise ValueError("unexpected or missing Parquet files under data/")
        split_rows = {
            s: sum(f["rows"] for f in files if f["kind"] == "logged" and f["split"] == s)
            for s in SPLITS
        }
        if sum(split_rows.values()) != config.rows:
            raise ValueError("final row count is inconsistent")
        atomic_json(
            root / "manifest.json",
            {
                "complete": True,
                "fingerprint": config.fingerprint,
                "rows": config.rows,
                "buildings": config.buildings,
                "split_rows": split_rows,
                "bytes": sum(f["bytes"] for f in files),
                "files": sorted(files, key=lambda f: f["path"]),
            },
        )
        write_card(root)
        atomic_json(
            root / "_SUCCESS.json",
            {
                "manifest_sha256": sha256(root / "manifest.json"),
                "metadata_sha256": metadata_hashes(root),
            },
        )
        return read_json(root / "manifest.json")
