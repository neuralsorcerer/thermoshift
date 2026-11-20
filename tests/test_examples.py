# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import subprocess
import sys
from pathlib import Path

import pytest

from thermoshift import validate

ROOT = Path(__file__).resolve().parents[1]


def run_example(script, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / script), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("script", "directory_option", "report_name"),
    [
        ("benchmarks/generation.py", "--output", "run_config.json"),
        ("examples/evaluate_policy.py", "--data", "manifest.json"),
        ("examples/train_baseline.py", "--data", "VALIDATION.JSON"),
    ],
)
def test_example_rejects_managed_report_destination(
    tmp_path, script, directory_option, report_name
):
    release = tmp_path / "release"
    result = run_example(script, directory_option, release, "--report", release / report_name)
    assert result.returncode == 2
    assert "experiment report must use a path separate from dataset files" in result.stderr
    assert not release.exists()


@pytest.mark.parametrize("option", ["--workers", "--batch-buildings"])
def test_benchmark_rejects_invalid_scheduling_before_creating_output(tmp_path, option):
    release = tmp_path / "release"
    result = run_example("benchmarks/generation.py", "--output", release, "--rows", 11, option, 0)
    assert result.returncode == 2
    assert "must be an integer" in result.stderr
    assert not release.exists()


@pytest.mark.parametrize("existing_directory", [False, True])
def test_benchmark_rejects_report_directory_before_creating_release(tmp_path, existing_directory):
    report = tmp_path / "reports"
    if existing_directory:
        report.mkdir()
        release = tmp_path / "release"
    else:
        release = report / "release"
    result = run_example(
        "benchmarks/generation.py", "--output", release, "--rows", 11, "--report", report
    )
    assert result.returncode == 2
    assert "experiment report must name a file" in result.stderr
    assert not release.exists()


def test_benchmark_writes_a_separate_report_inside_the_release(tmp_path):
    release = tmp_path / "release"
    report = release / "benchmark.json"
    result = run_example(
        "benchmarks/generation.py", "--output", release, "--rows", 11, "--report", report
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text(encoding="utf-8")) == json.loads(result.stdout)
    assert validate(release, replay=True, write_report=False)["rows"] == 11
