# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Verify distribution contents and exercise an installed wheel in a temporary target."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHECKOUT = """
from thermoshift._version import __version__
from thermoshift.provenance import source_digest

print(__version__)
print(source_digest())
"""

SMOKE = """
import json
import sys
from pathlib import Path

import thermoshift
from thermoshift import Config, generate, initialize, validate
from thermoshift.provenance import source_digest

def main():
    target, output, expected = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    if not Path(thermoshift.__file__).resolve().is_relative_to(target.resolve()):
        raise ValueError('Smoke test did not import the installed wheel')
    if source_digest() != expected:
        raise ValueError('Installed source fingerprint differs from the source tree')
    initialize(output, Config(rows=101, episode_steps=8, shard_buildings=7))
    generate(output, workers=2, batch_buildings=3)
    report = validate(output, batch_size=5, replay=True)
    if report['status'] != 'passed' or report['rows'] != 101 or not report['replay']:
        raise ValueError('Installed-wheel replay did not validate the expected rows')
    print(json.dumps({'version': thermoshift.__version__, 'rows': report['rows'],
                      'replay': report['replay'], 'source_sha256': source_digest()}))


if __name__ == '__main__':
    main()
"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _run(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SystemExit(detail or f"{Path(args[0]).name} exited with status {result.returncode}")
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.stdout


def _check_metadata(data: bytes, artifact: str, version: str, requires_python: str) -> None:
    metadata = BytesParser().parsebytes(data)
    _require(metadata["Name"] == "thermoshift", f"{artifact} metadata has an unexpected name")
    _require(metadata["Version"] == version, f"{artifact} metadata has an unexpected version")
    _require(
        metadata["Requires-Python"] == requires_python,
        f"{artifact} Python requirement differs from pyproject.toml",
    )
    _require(bool(metadata.get_all("Project-URL")), f"{artifact} metadata declares no project URLs")
    classifiers = metadata.get_all("Classifier") or []
    _require("Typing :: Typed" in classifiers, f"{artifact} metadata lacks the typed classifier")
    # PEP 639 forbids License classifiers beside a license expression.
    _require(
        metadata["License-Expression"] == "MIT", f"{artifact} has an unexpected license expression"
    )
    _require(
        not any(classifier.startswith("License ::") for classifier in classifiers),
        f"{artifact} metadata declares both a license expression and a license classifier",
    )


def main() -> None:
    wheels = list((ROOT / "dist").glob("*.whl"))
    archives = list((ROOT / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(archives) != 1:
        raise SystemExit("Build one wheel and one source archive with python -m build.")
    wheel, source_archive = wheels[0], archives[0]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    package = ROOT / "src/thermoshift"
    version, expected_digest = _run(
        [sys.executable, "-P", "-c", CHECKOUT],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(package.parent)},
    ).splitlines()
    package_paths = {
        path.relative_to(package).as_posix(): path
        for path in package.rglob("*")
        if path.is_file() and path.suffix in {".py", ".md"}
    }
    with zipfile.ZipFile(wheel) as archive:
        corrupt_member = archive.testzip()
        _require(corrupt_member is None, f"Wheel contains a corrupt member {corrupt_member}")
        members = {
            name.removeprefix("thermoshift/")
            for name in archive.namelist()
            if name.startswith("thermoshift/") and Path(name).suffix in {".py", ".md"}
        }
        _require(
            members == package_paths.keys(), "Wheel contains missing or unexpected package files"
        )
        for name, path in package_paths.items():
            _require(
                archive.read("thermoshift/" + name) == path.read_bytes(),
                f"Wheel copy of {name} differs from source",
            )
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        # The PEP 561 marker is useless unless it ships beside the package.
        _require("thermoshift/py.typed" in archive.namelist(), "Wheel is missing py.typed")
        _check_metadata(archive.read(metadata_path), "Wheel", version, project["requires-python"])
        license_path = metadata_path.removesuffix("METADATA") + "licenses/LICENSE"
        _require(
            license_path in archive.namelist()
            and archive.read(license_path) == (ROOT / "LICENSE").read_bytes(),
            "Wheel license file is missing or differs from source",
        )
        entrypoints = next(
            name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")
        )
        _require(
            "thermoshift = thermoshift.cli:main" in archive.read(entrypoints).decode(),
            "Wheel does not declare the thermoshift console entry point",
        )
    with tarfile.open(source_archive) as archive:
        prefix = archive.getnames()[0].split("/")[0]
        _check_metadata(
            archive.extractfile(f"{prefix}/PKG-INFO").read(),
            "Source archive",
            version,
            project["requires-python"],
        )
        source_paths = {"src/thermoshift/" + name: path for name, path in package_paths.items()}
        content_types = {
            "docs": {".py", ".md"},
            "examples": {".py"},
            "tests": {".py"},
            "scripts": {".py"},
            "benchmarks": {".py", ".md"},
            "notebooks": {".ipynb"},
        }
        for directory, suffixes in content_types.items():
            source_paths.update(
                {
                    path.relative_to(ROOT).as_posix(): path
                    for path in (ROOT / directory).rglob("*")
                    if path.is_file()
                    and path.suffix in suffixes
                    and not path.is_relative_to(ROOT / "docs/_build")
                    and ".ipynb_checkpoints" not in path.relative_to(ROOT).parts
                }
            )
        members = {
            name.removeprefix(prefix + "/")
            for name in archive.getnames()
            if name.startswith(
                tuple(
                    prefix + "/" + directory + "/"
                    for directory in (*content_types, "src/thermoshift")
                )
            )
            and Path(name).suffix in {".py", ".md", ".ipynb"}
        }
        _require(
            members == source_paths.keys(), "Source archive contains missing or unexpected files"
        )
        for name, path in source_paths.items():
            _require(
                archive.extractfile(f"{prefix}/{name}").read() == path.read_bytes(),
                f"Source archive copy of {name} differs from source",
            )
        for name in (
            "pyproject.toml",
            "README.md",
            "CONTRIBUTING.md",
            "constraints.txt",
            "MANIFEST.in",
            "LICENSE",
            "docs/requirements.txt",
            # CONTRIBUTING ships in the archive and tells the reader to run
            # pre-commit, so its configuration has to travel with it.
            ".pre-commit-config.yaml",
            "src/thermoshift/py.typed",
        ):
            _require(
                archive.extractfile(f"{prefix}/{name}").read() == (ROOT / name).read_bytes(),
                f"Source archive copy of {name} differs from source",
            )

    with tempfile.TemporaryDirectory(prefix="thermoshift-distribution-") as temporary:
        root = Path(temporary)
        target = root / "installed"
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--no-compile",
                "--disable-pip-version-check",
                "--target",
                str(target),
                str(wheel),
            ],
            cwd=root,
        )
        environment = {**os.environ, "PYTHONPATH": str(target)}
        smoke_path = root / "check_installed.py"
        smoke_path.write_text(SMOKE, encoding="utf-8")
        result = json.loads(
            _run(
                [
                    sys.executable,
                    str(smoke_path),
                    str(target),
                    str(root / "dataset"),
                    expected_digest,
                ],
                cwd=root,
                env=environment,
            )
        )
        cli_version = _run(
            [sys.executable, "-m", "thermoshift", "--version"],
            cwd=root,
            env=environment,
        )
        _require(
            cli_version.strip() == f"thermoshift {version}",
            "Installed-wheel CLI reports an unexpected version",
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "wheel": wheel.name,
                "source_archive": source_archive.name,
                "installed_wheel": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
