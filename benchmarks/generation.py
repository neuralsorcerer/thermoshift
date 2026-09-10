# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Measure dataset generation, validation, disk usage and parent-process memory."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

from thermoshift import Config, generate, initialize, validate
from thermoshift.filesystem import atomic_json, read_json
from thermoshift.provenance import runtime, source_digest


def benchmark(output: str | Path, config: Config, workers: int, batch_buildings: int) -> dict:
    """Generate a fresh release and return measurements for this run."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("benchmark output must be an empty directory")
    initialize(output, config)
    started = time.perf_counter()
    generation = generate(output, workers=workers, batch_buildings=batch_buildings)
    generation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    validation = validate(output)
    validation_seconds = time.perf_counter() - started
    manifest = read_json(output / "manifest.json")
    result = {
        "config": asdict(config),
        "runtime": runtime(),
        "platform": platform.platform(),
        "source_sha256": source_digest(),
        "workers": workers,
        "batch_buildings": batch_buildings,
        "decision_rows": config.rows,
        "buildings": config.buildings,
        "parquet_files": len(manifest["files"]),
        "parquet_bytes": manifest["bytes"],
        "bytes_per_decision": manifest["bytes"] / config.rows,
        "generation_seconds": generation_seconds,
        "validation_seconds": validation_seconds,
        "decisions_per_second": config.rows / generation_seconds,
        "generation": generation,
        "validation_status": validation["status"],
        "split_profiles": validation["split_profiles"],
    }
    try:
        import resource
    except ImportError:
        result["parent_peak_rss_mib"] = None
    else:
        units = 1 if platform.system() == "Darwin" else 1024
        result["parent_peak_rss_mib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * units / 2**20
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode-steps", type=int, default=168)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-buildings", type=int, default=512)
    parser.add_argument("--shard-buildings", type=int, default=16_384)
    parser.add_argument("--report")
    args = parser.parse_args()
    config = Config(
        rows=args.rows,
        seed=args.seed,
        episode_steps=args.episode_steps,
        shard_buildings=args.shard_buildings,
    )
    result = benchmark(args.output, config, args.workers, args.batch_buildings)
    if args.report:
        atomic_json(args.report, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
