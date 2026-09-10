# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Validate release integrity, scientific records and deterministic replay."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from thermoshift._checks import array, require
from thermoshift._version import __version__
from thermoshift.config import boolean, integer
from thermoshift.filesystem import atomic_json, sha256
from thermoshift.lifecycle import dataset_lock
from thermoshift.provenance import load_config, source_digest
from thermoshift.reading import _iter_pairs, _validated_manifest
from thermoshift.simulator import simulate
from thermoshift.validation.records import scan_records

__all__ = ["validate"]


def validate(
    root: str | Path,
    full: bool = True,
    batch_size: int = 65_536,
    write_report: bool = True,
    replay: bool = False,
) -> dict[str, Any]:
    """Validate a finalized dataset under an exclusive snapshot lease.

    Full validation checks record semantics and equations. Replay additionally
    regenerates each building with the initialized source and runtime. Set
    write_report=False for a read-only scan. A failed check raises ValueError."""
    for name, value in (("full", full), ("write_report", write_report), ("replay", replay)):
        boolean(value, name)
    integer(batch_size, "batch_size", 1)
    require(not replay or full, "replay requires full validation")
    with dataset_lock(root):
        return _validate_locked(root, full, batch_size, write_report, replay)


def _validate_locked(root, full=True, batch_size=65_536, write_report=True, replay=False):
    """Never leave a cached success after a failed or interrupted validation."""
    root = Path(root)
    started = time.perf_counter()
    status = {
        "status": "running",
        "scope": "full" if full else "checksums_and_metadata_only",
        "replay": replay,
        "validator_version": __version__,
        "validator_source_sha256": source_digest(),
    }
    if write_report:
        atomic_json(root / "validation.json", status)
    try:
        report = _validate_contents_locked(root, full, batch_size, replay)
    except BaseException as error:
        if write_report:
            atomic_json(
                root / "validation.json",
                {
                    **status,
                    "status": "failed" if isinstance(error, Exception) else "interrupted",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "seconds": time.perf_counter() - started,
                },
            )
        raise
    report.update(
        {
            "validator_version": __version__,
            "validator_source_sha256": status["validator_source_sha256"],
        }
    )
    if write_report:
        atomic_json(root / "validation.json", report)
    return report


def _validate_contents_locked(root, full=True, batch_size=65_536, replay=False):
    root = Path(root)
    started = time.perf_counter()
    config, manifest = _validated_manifest(root, verify_hash=True)
    report = {
        "status": "passed",
        "scope": "full" if full else "checksums_and_metadata_only",
        "fingerprint": config.fingerprint,
        "manifest_sha256": sha256(root / "manifest.json"),
        "rows": config.rows,
        "buildings": config.buildings,
        "files": len(manifest["files"]),
        "bytes": manifest["bytes"],
        "findings": [],
        "split_profiles": {},
        "replay": replay,
    }
    if full:
        report.update(scan_records(root, config, manifest, batch_size))
    if replay:
        # Reproducibility check, separate from the independent equation checks.
        load_config(root, check_runtime=True)
        replayed = 0
        for _, x, o in _iter_pairs(root, manifest, batch_size=config.episode_steps):
            bid = int(array(x, "building_id")[0])
            expected_x, expected_o, _ = simulate(config, bid, bid + 1)
            require(
                x.equals(expected_x) and o.equals(expected_o),
                f"deterministic replay differs at building {bid}",
            )
            replayed += x.num_rows
        require(replayed == config.rows, "replay row coverage mismatch")
        report["checks"].append("deterministic building replay with matching source/runtime")
    report["seconds"] = time.perf_counter() - started
    return report
