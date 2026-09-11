# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run configuration, source fingerprints and release metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import re
from dataclasses import fields
from pathlib import Path

from thermoshift._version import __version__
from thermoshift.config import Config, boolean
from thermoshift.filesystem import atomic_json, read_json, sha256
from thermoshift.schema import feature_roles, schema_document

BASE_METADATA = (
    "run_config.json",
    "schema.json",
    "feature_roles.json",
    "README.md",
    "DATASHEET.md",
    "LICENSE",
)
SOURCE_ROOT = Path(__file__).parent


def source_digest() -> str:
    """Fingerprint package code and dataset templates in a stable path order."""
    digest = hashlib.sha256()
    sources = sorted(path for path in SOURCE_ROOT.rglob("*") if path.suffix in {".py", ".md"})
    for path in sources:
        digest.update(path.relative_to(SOURCE_ROOT).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def runtime() -> dict[str, str]:
    """Versions used to initialize and resume a generation run."""
    return {
        "python": platform.python_version(),
        "thermoshift": __version__,
        **{name: importlib.metadata.version(name) for name in ("numpy", "pyarrow")},
    }


def metadata_hashes(root: str | Path) -> dict[str, str]:
    """Hash the release metadata bound by the completion marker."""
    return {name: sha256(Path(root) / name) for name in BASE_METADATA}


def check_schema_documents(root: str | Path, repair_missing: bool = False) -> None:
    """Verify schema and feature documents, optionally creating missing copies."""
    for name, expected in (
        ("schema.json", schema_document()),
        ("feature_roles.json", feature_roles()),
    ):
        path = Path(root) / name
        if repair_missing and not path.exists():
            atomic_json(path, expected)
        if read_json(path) != expected:
            raise ValueError(f"{name} differs from this schema version")


def load_config(root: str | Path, check_runtime: bool = False) -> Config:
    """Load a configuration and verify its fingerprint and provenance structure."""
    boolean(check_runtime, "check_runtime")
    document = read_json(Path(root) / "run_config.json")
    required = {"config", "fingerprint", "source_sha256", "runtime"}
    if not isinstance(document, dict) or not required.issubset(document):
        raise ValueError("run configuration is missing required provenance fields")
    settings = document["config"]
    if not isinstance(settings, dict) or set(settings) != {f.name for f in fields(Config)}:
        raise ValueError("run configuration must contain every Config field")
    digest = document["source_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid source SHA-256 in run configuration")
    versions = document["runtime"]
    required_versions = {"python", "numpy", "pyarrow"}
    if (
        not isinstance(versions, dict)
        or not required_versions.issubset(versions)
        # Preserve read access to immutable releases made before the package rename.
        or not {"thermoshift", "thermoshift-synth"}.intersection(versions)
        or any(not isinstance(v, str) or not v for v in versions.values())
    ):
        raise ValueError("invalid runtime provenance in run configuration")
    config = Config(**settings)
    if document["fingerprint"] != config.fingerprint:
        raise ValueError("configuration fingerprint mismatch")
    if check_runtime and (
        document["source_sha256"] != source_digest() or document["runtime"] != runtime()
    ):
        raise ValueError(
            "generator code/runtime differs from initialization; restore the original environment or use a new output directory"
        )
    return config
