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
import sys
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
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
from thermoshift.filesystem import (
    atomic_json,
    dataset_parquet_paths,
    dataset_path,
    fsync_directory,
    is_redirected,
    read_json,
    require_regular_file,
    sha256,
)
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


def _sync_dataset_root_parents(root: Path) -> None:
    """Flush the new dataset's path entries on its containing filesystem."""
    if os.name != "posix":
        return
    root = root.resolve()
    device = root.stat().st_dev
    for parent in root.parents:
        if parent.stat().st_dev != device:
            break
        try:
            fsync_directory(parent)
        except PermissionError:
            # A caller's parent directory can allow traversal without permitting an open.
            continue


def initialize(root: str | Path, config: Config) -> Config:
    """Initialize an empty directory or reopen a matching generation plan."""
    if not isinstance(config, Config):
        raise TypeError("config must be a Config instance")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with dataset_lock(root):
        path = dataset_path(root, "run_config.json")
        if path.exists():
            current = load_config(root, check_runtime=True)
            if current.fingerprint != config.fingerprint:
                raise ValueError(
                    "output contains a different configuration; use a new output directory"
                )
            check_schema_documents(root, repair_missing=True)
            # The last schema rename may have succeeded before its directory sync failed.
            fsync_directory(root)
            return config
        # A process may have died before its first atomic config rename.
        for temporary in root.glob("run_config.json.tmp-*"):
            temporary.unlink()
        if any(p.name != ".lifecycle.lock" for p in root.iterdir()):
            raise ValueError("new dataset output directory must be empty")
        _sync_dataset_root_parents(root)
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


def _invalidate_release(root: Path) -> None:
    """Durably clear release certification before changing shard or manifest data."""
    (root / "validation.json").unlink(missing_ok=True)
    (root / "_SUCCESS.json").unlink(missing_ok=True)
    # A different generation rank may have performed the unlinks but not synced yet.
    fsync_directory(root)


def _generate_shard_locked(root, config_dict, shard_id, batch_buildings):
    config = Config(**config_dict)
    root = Path(root)
    state = dataset_path(root, "_state")
    state.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lock_path = dataset_path(root, f"_state/shard-{shard_id:08d}.lock")
    require_regular_file(lock_path, missing_ok=True)
    with FileLock(str(lock_path), timeout=0):
        relative_state = state_path(".", shard_id)
        checkpoint = dataset_path(root, relative_state)
        require_regular_file(checkpoint, missing_ok=True)
        if shard_complete(root, config, shard_id):
            remove_temporaries(root, shard_id)
            # A completed checkpoint rename may still be waiting for a directory sync.
            fsync_directory(state)
            return {
                "shard_id": shard_id,
                "status": "resumed",
                "seconds": time.perf_counter() - started,
            }
        if checkpoint.exists():
            checkpoint.unlink()
            fsync_directory(state)
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
                        selected = codes == split_id
                        if not np.any(selected):
                            continue
                        mask = pa.array(selected)
                        for kind, table in (("logged", logged), ("oracle", oracle)):
                            relative = shard_path(kind, split, shard_id)
                            if relative not in writers:
                                final = dataset_path(root, relative)
                                require_regular_file(final, missing_ok=True)
                                final.parent.mkdir(parents=True, exist_ok=True)
                                temporary = dataset_path(
                                    root, Path(relative).with_suffix(".parquet.inprogress")
                                )
                                # Start with a new inode for an interrupted shard.
                                temporary.unlink(missing_ok=True)
                                handle = resources.enter_context(temporary.open("xb"))
                                temp_paths[relative] = temporary
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
                temporary = dataset_path(root, temporary.relative_to(root))
                with temporary.open("r+b") as handle:
                    os.fsync(handle.fileno())
                # Re-resolve the destination: nothing guarantees a path stayed
                # symlink-free between opening the temporary and committing it.
                final = dataset_path(root, relative)
                os.replace(temporary, final)
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
            # Sync the whole path to each partition, including directory entries
            # created by other ranks. The root sync also persists the _state entry.
            directories = {
                directory
                for path in temp_paths.values()
                for directory in path.parents
                if directory.is_relative_to(root)
            }
            for directory in sorted(directories, key=lambda path: -len(path.parts)):
                fsync_directory(directory)
            atomic_json(
                dataset_path(root, relative_state),
                {
                    "fingerprint": config.fingerprint,
                    "shard_id": shard_id,
                    "provenance_sha256": sha256(root / "run_config.json"),
                    "files": files,
                },
            )
        finally:
            for path in temp_paths.values():
                try:
                    temporary = dataset_path(root, path.relative_to(root))
                except ValueError:
                    # A redirected parent no longer names the file we opened.
                    continue
                temporary.unlink(missing_ok=True)
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
        if rank < config.shards:
            _invalidate_release(root)
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
    if sys.platform == "win32":
        workers = min(workers, 61)  # ProcessPoolExecutor's Windows limit.
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
        try:
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
        except BrokenProcessPool as error:
            # A worker that raises reports its own exception through the future.
            # Reaching here means one died without reporting anything, so name
            # the usual cause and the remedy instead of an opaque pool failure.
            raise OSError(
                "a generation worker exited without reporting an error, which usually means "
                "the operating system stopped it for memory use; reduce workers or "
                "batch_buildings and run generation again to resume completed shards"
            ) from error
    summary["seconds"] = time.perf_counter() - started
    return summary


def finalize(root: str | Path) -> dict[str, Any]:
    """Verify all committed shards and write the manifest, card and completion marker."""
    root = Path(root)
    with dataset_lock(root):
        config = load_config(root, check_runtime=True)
        # A failed or interrupted attempt must not leave a previous completion
        # marker or scientific report certifying the changed release.
        _invalidate_release(root)
        check_schema_documents(root)
        actual = dataset_parquet_paths(root)
        # Reject a redirected shard-state directory before trusting its records.
        state = dataset_path(root, "_state")
        if state.is_dir() and any(is_redirected(path) for path in state.iterdir()):
            raise ValueError("symlinks and junctions are not allowed under _state/")
        files = []
        for sid in range(config.shards):
            if not shard_complete(root, config, sid, verify_hash=True):
                raise ValueError(f"shard {sid} is incomplete; generate all ranks before finalizing")
            remove_temporaries(root, sid)
            files.extend(read_json(state_path(root, sid))["files"])
        expected = {f["path"] for f in files}
        if actual != expected:
            raise ValueError("unexpected or missing Parquet files under data/")
        split_rows = {
            s: sum(f["rows"] for f in files if f["kind"] == "logged" and f["split"] == s)
            for s in SPLITS
        }
        if sum(split_rows.values()) != config.rows:
            raise ValueError("final row count is inconsistent")
        fsync_directory(state)
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
