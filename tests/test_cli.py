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
