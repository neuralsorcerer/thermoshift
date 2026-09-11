# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Render the column reference from the package's Arrow schemas."""

from __future__ import annotations

import argparse
from pathlib import Path

from thermoshift.schema import schema_document

ROOT = Path(__file__).resolve().parents[1]


def render() -> str:
    lines = [
        "# Column reference",
        "",
        "[Project overview](index.md) · [Python API](api.md)",
        "",
        "Columns are defined in `src/thermoshift/schema.py`. Each configuration has one row per",
        "decision, joined by `row_id`. The `role` field identifies how a column is used.",
        "",
    ]
    for kind, fields in schema_document().items():
        lines.extend(
            [
                f"## {kind}",
                "",
                "| Column | Type | Unit | Role | Nullable | Description |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for field in fields:
            dtype = {"float": "float32", "double": "float64"}.get(field["dtype"], field["dtype"])
            nullable = "yes" if field["nullable"] else "no"
            description = field["description"].replace("|", "\\|")
            lines.append(
                f"| `{field['name']}` | {dtype} | {field['unit']} | {field['role']} | "
                f"{nullable} | {description} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "docs/schema.md"
    expected = render()
    if args.check:
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise SystemExit("Column reference is stale; run python scripts/update_schema.py")
        print("Column reference matches the package schema.")
    else:
        path.write_text(expected, encoding="utf-8")
        print("Updated docs/schema.md")


if __name__ == "__main__":
    main()
