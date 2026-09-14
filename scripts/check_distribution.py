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
import zipfile
from email.parser import BytesParser
from pathlib import Path

from thermoshift._version import __version__
from thermoshift.provenance import source_digest

ROOT = Path(__file__).resolve().parents[1]

SMOKE = """
import json
import sys
from pathlib import Path

import thermoshift
from thermoshift import Config, generate, initialize, validate
from thermoshift.provenance import source_digest

def main():
    target, output, expected = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    assert Path(thermoshift.__file__).resolve().is_relative_to(target.resolve())
    assert source_digest() == expected
    initialize(output, Config(rows=101, episode_steps=8, shard_buildings=7))
    generate(output, workers=2, batch_buildings=3)
    report = validate(output, batch_size=5, replay=True)
    assert report['status'] == 'passed' and report['rows'] == 101
    print(json.dumps({'version': thermoshift.__version__, 'rows': report['rows'],
                      'replay': report['replay'], 'source_sha256': source_digest()}))


if __name__ == '__main__':
    main()
"""


def main() -> None:
    wheels = list((ROOT / "dist").glob("*.whl"))
    archives = list((ROOT / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(archives) != 1:
        raise SystemExit("Build one wheel and one source archive with python -m build.")
    wheel, source_archive = wheels[0], archives[0]
    package = ROOT / "src/thermoshift"
    package_paths = {
        path.relative_to(package).as_posix(): path
        for path in package.rglob("*")
        if path.is_file() and path.suffix in {".py", ".md"}
    }
    with zipfile.ZipFile(wheel) as archive:
        assert archive.testzip() is None
        members = {
            name.removeprefix("thermoshift/")
            for name in archive.namelist()
            if name.startswith("thermoshift/") and Path(name).suffix in {".py", ".md"}
        }
        assert members == package_paths.keys(), "Wheel contains missing or unexpected package files"
        for name, path in package_paths.items():
            assert archive.read("thermoshift/" + name) == path.read_bytes()
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        # The PEP 561 marker is useless unless it ships beside the package.
        assert "thermoshift/py.typed" in archive.namelist(), "Wheel is missing py.typed"
        metadata = BytesParser().parsebytes(archive.read(metadata_path))
        assert metadata["Name"] == "thermoshift"
        assert metadata["Version"] == __version__
        assert metadata.get_all("Project-URL"), "Wheel metadata declares no project URLs"
        classifiers = metadata.get_all("Classifier") or []
        assert "Typing :: Typed" in classifiers
        # PEP 639 forbids License classifiers beside a license expression.
        assert metadata["License-Expression"] == "MIT"
        assert not [c for c in classifiers if c.startswith("License ::")]
        entrypoints = next(
            name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")
        )
        assert "thermoshift = thermoshift.cli:main" in archive.read(entrypoints).decode()
    with tarfile.open(source_archive) as archive:
        prefix = archive.getnames()[0].split("/")[0]
        source_paths = {"src/thermoshift/" + name: path for name, path in package_paths.items()}
        content_directories = ("docs", "examples", "tests", "scripts", "notebooks", "benchmarks")
        for directory in content_directories:
            source_paths.update(
                {
                    path.relative_to(ROOT).as_posix(): path
                    for path in (ROOT / directory).rglob("*")
                    if path.is_file() and path.suffix in {".py", ".md", ".ipynb"}
                }
            )
        members = {
            name.removeprefix(prefix + "/")
            for name in archive.getnames()
            if name.startswith(
                tuple(
                    prefix + "/" + directory + "/"
                    for directory in (*content_directories, "src/thermoshift")
                )
            )
            and Path(name).suffix in {".py", ".md", ".ipynb"}
        }
        assert members == source_paths.keys(), "Source archive contains missing or unexpected files"
        for name, path in source_paths.items():
            assert archive.extractfile(f"{prefix}/{name}").read() == path.read_bytes()
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
            assert archive.extractfile(f"{prefix}/{name}").read() == (ROOT / name).read_bytes()

    with tempfile.TemporaryDirectory(prefix="thermoshift-distribution-") as temporary:
        root = Path(temporary)
        target = root / "installed"
        subprocess.run(
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
            check=True,
            capture_output=True,
            text=True,
        )
        environment = {**os.environ, "PYTHONPATH": str(target)}
        smoke_path = root / "check_installed.py"
        smoke_path.write_text(SMOKE, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(smoke_path), str(target), str(root / "dataset"), source_digest()],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        version = subprocess.run(
            [sys.executable, "-m", "thermoshift", "--version"],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert version.stdout.strip() == f"thermoshift {__version__}"
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
