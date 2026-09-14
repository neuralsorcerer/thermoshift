# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Command workflows, output channels, provenance coverage and worker allocation."""

import json
from pathlib import Path

import pytest

from thermoshift import Config, __version__, generate, initialize
from thermoshift.cli import main


def invoke(capsys, *arguments):
    status = main(list(arguments))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def test_cli_workflow_emits_one_json_result_and_separate_progress(tmp_path, capsys):
    root = str(tmp_path / "dataset")
    status, output, error = invoke(capsys, "init", root, "--rows", "73", "--episode-steps", "8")
    assert status == 0 and json.loads(output)["rows"] == 73 and error == ""
    status, output, error = invoke(capsys, "generate", root)
    assert status == 0 and json.loads(output)["written"] == 1
    assert json.loads(error)["status"] == "written"
    status, output, error = invoke(capsys, "validate", root, "--replay")
    assert status == 0 and json.loads(output)["replay"] is True and error == ""


def test_cli_quiet_resume_and_readonly_validation(tmp_path, capsys):
    root = str(tmp_path / "dataset")
    invoke(capsys, "init", root, "--rows", "17", "--episode-steps", "8")
    status, output, error = invoke(capsys, "generate", root, "--quiet")
    assert status == 0 and json.loads(output)["written"] == 1 and error == ""
    status, output, error = invoke(capsys, "generate", root, "--quiet")
    assert status == 0 and json.loads(output)["resumed"] == 1 and error == ""
    invoke(capsys, "validate", root)
    before = (Path(root) / "validation.json").read_bytes()
    status, output, error = invoke(capsys, "validate", root, "--metadata-only", "--no-report")
    assert status == 0 and json.loads(output)["scope"] == "checksums_and_metadata_only"
    assert (Path(root) / "validation.json").read_bytes() == before


def test_invalid_cli_configuration_returns_error_without_output(tmp_path, capsys):
    status, output, error = invoke(capsys, "init", str(tmp_path / "invalid"), "--rows", "0")
    assert status == 2 and output == "" and "rows must" in error
    assert not (tmp_path / "invalid").exists()


def test_cli_version_uses_package_version(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--version"])
    assert stopped.value.code == 0
    assert capsys.readouterr().out.strip() == f"thermoshift {__version__}"


def test_finalize_and_publish_plan_reach_their_command_branches(tmp_path, capsys):
    root = str(tmp_path / "dataset")
    invoke(capsys, "init", root, "--rows", "48", "--episode-steps", "8")
    invoke(capsys, "generate", root, "--quiet")
    status, output, error = invoke(capsys, "finalize", root)
    assert status == 0 and error == ""
    assert set(json.loads(output)) == {"rows", "files", "bytes"}
    assert json.loads(output)["rows"] == 48
    status, output, error = invoke(capsys, "publish", root, "--repo-id", "owner/set", "--dry-run")
    plan = json.loads(output)
    assert status == 0
    assert plan["dry_run"] is True and plan["repo_id"] == "owner/set"
    assert plan["visibility"] == "private"
    # A dry run plans only; it never reaches a commit.
    assert "commit_sha" not in plan and "url" not in plan


def test_plan_reports_the_documented_billion_decision_layout(tmp_path, capsys):
    # The figures quoted in README.md and docs/generation.md for the default
    # episode and shard sizes at one billion decisions.
    root = str(tmp_path / "dataset")
    invoke(capsys, "init", root, "--rows", "1000000000")
    status, output, error = invoke(capsys, "plan", root, "--batch-buildings", "256")
    plan = json.loads(output)
    assert status == 0 and error == ""
    assert plan == {
        "rows": 1_000_000_000,
        "buildings": 5_952_381,
        "shards": 364,
        "parquet_files_upper_bound": 3_640,
        "simulation_rows_per_batch_upper_bound": 256 * 168,
    }
    status, _, error = invoke(capsys, "plan", root, "--batch-buildings", "0")
    assert status == 2 and "batch_buildings" in error


def test_interruption_returns_the_documented_status(tmp_path, capsys, monkeypatch):
    from thermoshift import cli

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("injected cancellation")

    monkeypatch.setattr(cli, "initialize", interrupt)
    status, output, error = invoke(capsys, "init", str(tmp_path / "dataset"), "--rows", "8")
    assert status == 130 and output == "" and "interrupted" in error


def test_module_entrypoint_runs_the_same_command_line():
    import subprocess
    import sys

    from thermoshift.__main__ import main as module_main
    from thermoshift.cli import main as cli_main

    assert module_main is cli_main
    completed = subprocess.run(
        [sys.executable, "-m", "thermoshift", "--version"], capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == f"thermoshift {__version__}"


def test_source_fingerprint_covers_nested_modules_and_templates(tmp_path, monkeypatch):
    from thermoshift import provenance

    (tmp_path / "validation").mkdir()
    (tmp_path / "resources").mkdir()
    module = tmp_path / "validation/records.py"
    template = tmp_path / "resources/CARD.md"
    module.write_text("value = 1\n")
    template.write_text("Dataset card\n")
    monkeypatch.setattr(provenance, "SOURCE_ROOT", tmp_path)
    first = provenance.source_digest()
    module.write_text("value = 2\n")
    second = provenance.source_digest()
    template.write_text("Updated dataset card\n")
    assert len({first, second, provenance.source_digest()}) == 3


def test_source_fingerprint_orders_files_the_same_way_on_every_platform(tmp_path, monkeypatch):
    # Files are hashed in sequence, so the sequence must be identical on every
    # platform. Ordering Path objects is not: comparison walks path components, so
    # "sub.py" sorts after "sub/a.py" although the hashed names sort the other way,
    # and Windows also folds case.
    import hashlib
    from pathlib import PurePosixPath, PureWindowsPath

    from thermoshift import provenance

    (tmp_path / "sub").mkdir()
    contents = {
        "Utils.py": "u\n",
        "alpha.py": "a\n",
        "sub.py": "s\n",
        "sub/CARD.md": "c\n",
        "sub/a.py": "n\n",
    }
    for name, text in contents.items():
        (tmp_path / name).write_text(text)
    (tmp_path / "notes.txt").write_text("ignored\n")  # only .py and .md are covered
    monkeypatch.setattr(provenance, "SOURCE_ROOT", tmp_path)

    expected = hashlib.sha256()
    for name in sorted(contents):  # plain string order, the same everywhere
        expected.update(name.encode("utf-8") + b"\0")
        expected.update(hashlib.sha256((tmp_path / name).read_bytes()).digest())
    assert provenance.source_digest() == expected.hexdigest()

    # Both Path orderings disagree with that, so the assertion above fails if the
    # sort ever goes back to comparing Path objects, on this platform and on the
    # other one alike.
    names = sorted(contents)
    assert names != [p.as_posix() for p in sorted(PurePosixPath(n) for n in names)]
    assert names != [
        PurePosixPath(str(p).replace("\\", "/")).as_posix()
        for p in sorted(PureWindowsPath(n) for n in names)
    ]


def test_single_assigned_shard_avoids_process_pool(tmp_path, monkeypatch):
    from thermoshift import storage

    def forbidden(*args, **kwargs):
        raise AssertionError("single-shard generation should run in the caller")

    initialize(tmp_path, Config(rows=17, episode_steps=8))
    monkeypatch.setattr(storage, "ProcessPoolExecutor", forbidden)
    assert generate(tmp_path, workers=8)["written"] == 1


def test_empty_rank_returns_empty_summary(tmp_path):
    initialize(tmp_path, Config(rows=17, episode_steps=8))
    result = generate(tmp_path, workers=8, world_size=4, rank=3)
    assert result["written"] == result["resumed"] == 0
