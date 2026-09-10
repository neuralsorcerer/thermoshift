# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate Hugging Face dataset metadata and release documentation."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import yaml

from thermoshift.config import SPLITS
from thermoshift.filesystem import atomic_bytes, read_json
from thermoshift.provenance import load_config
from thermoshift.schema import SCHEMAS


def write_card(root: str | Path) -> None:
    """Write dataset configurations, the model reference and license for a finalized manifest."""
    root = Path(root)
    config = load_config(root)
    manifest = read_json(root / "manifest.json")
    front = {
        "pretty_name": "ThermoShift",
        "license": "mit",
        "task_categories": [
            "tabular-regression",
            "tabular-classification",
            "reinforcement-learning",
        ],
        "tags": [
            "synthetic",
            "energy",
            "causal-inference",
            "counterfactual",
            "offline-rl",
            "distribution-shift",
            "time-series",
        ],
        "configs": [],
    }
    for kind in SCHEMAS:
        front["configs"].append(
            {
                "config_name": kind,
                "default": kind == "logged",
                "data_files": [
                    {"split": split, "path": f"data/{kind}/{split}/**/*.parquet"}
                    for split in SPLITS
                    if manifest["split_rows"][split] > 0
                ],
            }
        )
    body = files("thermoshift").joinpath("resources/CARD.md").read_text(encoding="utf-8")
    summary = f"\n## This release\n\n{config.rows:,} unique hourly decision records from {config.buildings:,} synthetic buildings. "
    summary += "Each decision has a matching row in the logged and oracle configurations, linked by row_id.\n\n"
    summary += "| Split | Decision records |\n| --- | ---: |\n"
    summary += "\n".join(
        f"| {split} | {count:,} |" for split, count in manifest["split_rows"].items()
    )
    summary += f"\n\nSeed: `{config.seed}`. Episode length: `{config.episode_steps}` hourly steps. "
    summary += f"Hidden-confounding mode: `{config.hidden_confounding}`. Config fingerprint: `{config.fingerprint}`.\n"
    summary += f"\nCompressed Parquet bytes (both configurations): `{manifest['bytes']}`. "
    summary += "Generation parameters and runtime versions are in `run_config.json`. Column roles and units are in `schema.json`.\n"
    atomic_bytes(
        root / "README.md",
        ("---\n" + yaml.safe_dump(front, sort_keys=False) + "---\n\n" + body + summary).encode(),
    )
    atomic_bytes(
        root / "DATASHEET.md", files("thermoshift").joinpath("resources/DATASHEET.md").read_bytes()
    )
    atomic_bytes(
        root / "LICENSE", files("thermoshift").joinpath("resources/LICENSE.md").read_bytes()
    )
