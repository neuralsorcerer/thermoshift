# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Command-line workflows with JSON results and progress on standard error."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from thermoshift._version import __version__
from thermoshift.config import Config, integer
from thermoshift.provenance import load_config
from thermoshift.storage import finalize, generate, initialize


def parser() -> argparse.ArgumentParser:
    """Build the parser shared by console and module entrypoints."""
    root = argparse.ArgumentParser(
        prog="thermoshift",
        description="Generate, validate and publish synthetic cooling datasets.",
    )
    root.add_argument("--version", action="version", version=f"thermoshift {__version__}")
    commands = root.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Create a dataset configuration")
    init.add_argument("output", help="Dataset directory")
    init.add_argument("--rows", type=int, default=1_000_000)
    init.add_argument("--seed", type=int, default=42)
    init.add_argument("--episode-steps", type=int, default=168)
    init.add_argument("--shard-buildings", type=int, default=16_384)
    init.add_argument("--exploration", type=float, default=0.15)
    init.add_argument("--hidden-confounding", action="store_true")
    init.add_argument("--comfort-weight", type=float, default=0.30)
    init.add_argument("--carbon-weight", type=float, default=0.05)

    gen = commands.add_parser("generate", help="Generate or resume Parquet shards")
    gen.add_argument("output")
    gen.add_argument("--workers", type=int, default=1)
    gen.add_argument("--batch-buildings", type=int, default=512)
    gen.add_argument("--rank", type=int, default=0)
    gen.add_argument("--world-size", type=int, default=1)
    gen.add_argument("--quiet", action="store_true", help="Suppress per-shard progress")

    final = commands.add_parser("finalize", help="Finalize after every generation rank finishes")
    final.add_argument("output")

    validate = commands.add_parser(
        "validate", help="Check release integrity and record consistency"
    )
    validate.add_argument("output")
    validate.add_argument("--batch-size", type=int, default=65_536)
    validate.add_argument(
        "--metadata-only", action="store_true", help="Check hashes, schemas and file counts"
    )
    validate.add_argument(
        "--replay", action="store_true", help="Regenerate and compare every building"
    )
    validate.add_argument(
        "--no-report", action="store_true", help="Leave validation.json unchanged"
    )

    publish = commands.add_parser("publish", help="Upload a finalized release to Hugging Face")
    publish.add_argument("output")
    publish.add_argument("--repo-id", required=True, help="OWNER/DATASET")
    publish.add_argument("--public", action="store_true", help="Use a public repository")
    publish.add_argument("--revision", default="main")
    publish.add_argument("--dry-run", action="store_true", help="Validate and show the upload plan")

    plan = commands.add_parser("plan", help="Show row, shard and batch counts")
    plan.add_argument("output")
    plan.add_argument("--batch-buildings", type=int, default=512)
    return root


def _progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _execute(command: str, output: str, options: dict[str, Any]) -> dict:
    if command == "init":
        config = initialize(output, Config(**options))
        return {
            "rows": config.rows,
            "buildings": config.buildings,
            "shards": config.shards,
            "fingerprint": config.fingerprint,
        }
    if command == "generate":
        progress = None if options.pop("quiet") else _progress
        return generate(output, progress=progress, **options)
    if command == "finalize":
        manifest = finalize(output)
        return {
            "rows": manifest["rows"],
            "files": len(manifest["files"]),
            "bytes": manifest["bytes"],
        }
    if command == "validate":
        from thermoshift.validation import validate

        return validate(
            output,
            full=not options["metadata_only"],
            batch_size=options["batch_size"],
            write_report=not options["no_report"],
            replay=options["replay"],
        )
    if command == "publish":
        from thermoshift.hub import publish

        return publish(output, **options)

    integer(options["batch_buildings"], "batch_buildings", 1)
    config = load_config(output)
    return {
        "rows": config.rows,
        "buildings": config.buildings,
        "shards": config.shards,
        "parquet_files_upper_bound": 10 * config.shards,
        "simulation_rows_per_batch_upper_bound": min(
            options["batch_buildings"], config.shard_buildings
        )
        * config.episode_steps,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command and return its process exit status."""
    options = vars(parser().parse_args(argv))
    command, output = options.pop("command"), options.pop("output")
    try:
        result = _execute(command, output, options)
    except (ValueError, OSError) as error:
        print(f"thermoshift: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("thermoshift: interrupted", file=sys.stderr)
        return 130
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
