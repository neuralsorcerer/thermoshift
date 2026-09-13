# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Failure recovery, API boundaries and resource cleanup."""

import multiprocessing
from pathlib import Path

import nbformat
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from thermoshift import Config
from thermoshift.filesystem import atomic_json, read_json, sha256
from thermoshift.hub import publish
from thermoshift.lifecycle import dataset_lock
from thermoshift.provenance import check_schema_documents, load_config, metadata_hashes
from thermoshift.reading import _iter_pairs, iter_pairs, validated_manifest
from thermoshift.schema import FEATURES, feature_roles
from thermoshift.shards import shard_complete, shard_path
from thermoshift.simulator import simulate
from thermoshift.storage import finalize, generate, initialize
from thermoshift.validation import validate


@pytest.fixture
def release(tmp_path):
    initialize(tmp_path, Config(rows=1501, episode_steps=24, shard_buildings=17))
    generate(tmp_path, batch_buildings=7, progress=None)
    return tmp_path


def resign(root):
    manifest = read_json(root / "manifest.json")
    for item in manifest["files"]:
        path = root / item["path"]
        item["sha256"], item["bytes"] = sha256(path), path.stat().st_size
    manifest["bytes"] = sum(f["bytes"] for f in manifest["files"])
    atomic_json(root / "manifest.json", manifest)
    atomic_json(
        root / "_SUCCESS.json",
        {
            "manifest_sha256": sha256(root / "manifest.json"),
            "metadata_sha256": metadata_hashes(root),
        },
    )


def test_feature_roles_cannot_mutate_global_feature_whitelist():
    original = FEATURES.copy()
    roles = feature_roles()
    roles["policy_features"].append("y_reward")
    roles["transition_features"].clear()
    assert FEATURES == original
    assert feature_roles()["policy_features"] == original
    assert "action" in feature_roles()["transition_features"]


@pytest.mark.parametrize("name", ["exploration", "comfort_weight", "carbon_weight"])
def test_oversized_integer_is_a_clear_configuration_error(name):
    with pytest.raises(ValueError, match="finite"):
        Config(**{name: 10**10000})


@pytest.mark.parametrize("name", ["public", "dry_run"])
@pytest.mark.parametrize("value", ["false", 0])
def test_publish_flags_rejected_before_validation_or_network(tmp_path, monkeypatch, name, value):
    from thermoshift import validation

    def forbidden(*args, **kwargs):
        raise AssertionError("validation should not run for invalid publication flags")

    monkeypatch.setattr(validation, "_validate_locked", forbidden)
    with pytest.raises(ValueError, match=name):
        publish(tmp_path / "does-not-exist", "test/release", **{name: value})


@pytest.mark.parametrize("name", ["full", "write_report", "replay"])
@pytest.mark.parametrize("value", ["false", 0])
def test_validation_flags_require_actual_booleans(tmp_path, name, value):
    with pytest.raises(ValueError, match=name):
        validate(tmp_path / "does-not-exist", **{name: value})


def test_generation_preserves_callers_arrow_pools(tmp_path):
    before = pa.cpu_count(), pa.io_thread_count()
    try:
        pa.set_cpu_count(7)
        pa.set_io_thread_count(3)
        initialize(tmp_path, Config(rows=29, episode_steps=8))
        generate(tmp_path, progress=None)
        assert (pa.cpu_count(), pa.io_thread_count()) == (7, 3)
    finally:
        pa.set_cpu_count(before[0])
        pa.set_io_thread_count(before[1])


def test_invalid_progress_preserves_completion(release):
    marker = (release / "_SUCCESS.json").read_bytes()
    with pytest.raises(ValueError, match="progress"):
        generate(release, progress="verbose")
    assert (release / "_SUCCESS.json").read_bytes() == marker


@pytest.mark.parametrize("write_report", [True, False])
def test_failed_validation_never_leaves_stale_success_unless_explicitly_readonly(
    release, write_report
):
    validate(release)
    original_report = (release / "validation.json").read_bytes()
    (release / "README.md").write_text("corrupted metadata", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata checksum"):
        validate(release, write_report=write_report)
    if write_report:
        report = read_json(release / "validation.json")
        assert report["status"] == "failed" and report["error_type"] == "ValueError"
        assert len(report["validator_source_sha256"]) == 64
    else:
        assert (release / "validation.json").read_bytes() == original_report


def test_interrupted_validation_records_interruption(release, monkeypatch):
    import thermoshift.validation as module

    validate(release)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("injected cancellation")

    monkeypatch.setattr(module, "_validated_manifest", interrupt)
    with pytest.raises(KeyboardInterrupt):
        validate(release)
    assert read_json(release / "validation.json")["status"] == "interrupted"


def test_symlinked_data_root_is_rejected(release):
    moved = release / "moved-data"
    (release / "data").rename(moved)
    (release / "data").symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validated_manifest(release)


def test_negative_requested_load_during_outage_is_rejected(release):
    for item in read_json(release / "manifest.json")["files"]:
        if item["kind"] != "logged":
            continue
        x = pq.read_table(release / item["path"])
        outage = np.flatnonzero(~x["grid_available"].to_numpy())
        if not len(outage):
            continue
        path = release / item["path"].replace("data/logged/", "data/oracle/")
        o = pq.read_table(path)
        values = o["base_load_kw"].to_numpy().copy()
        values[outage[0]] = -1
        i = o.schema.get_field_index("base_load_kw")
        o = o.set_column(i, o.schema.field(i), pa.array(values))
        pq.write_table(o, path)
        resign(release)
        with pytest.raises(ValueError, match="nonpositive base_load"):
            validate(release)
        return
    pytest.fail("deterministic fixture must contain an outage")


@pytest.mark.parametrize("field", ["source_sha256", "runtime", "config.seed", "config.rows"])
def test_missing_provenance_or_default_config_fields_are_rejected(release, field):
    document = read_json(release / "run_config.json")
    if field.startswith("config."):
        del document["config"][field.split(".")[1]]
    else:
        del document[field]
    atomic_json(release / "run_config.json", document)
    with pytest.raises(ValueError, match="provenance|Config field"):
        load_config(release)


def test_numeric_order_when_shard_filename_padding_width_is_exceeded(tmp_path):
    # Two sparse files per configuration test this boundary without generating
    # one hundred million shards. Both are deliberately assigned the same split.
    ids = [99_999_999, 100_000_000]
    config = Config(rows=(ids[-1] + 1) * 2, episode_steps=2, shard_buildings=1)
    entries = []
    for sid in ids:
        x, o, _ = simulate(config, sid, sid + 1)
        for kind, table in (("logged", x), ("oracle", o)):
            path = tmp_path / shard_path(kind, "test", sid)
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, path)
            entries.append(
                {
                    "kind": kind,
                    "split": "test",
                    "shard_id": sid,
                    "path": path.relative_to(tmp_path).as_posix(),
                }
            )
    manifest = {"files": sorted(entries, key=lambda f: f["path"])}
    seen = []
    for item, x, o in _iter_pairs(tmp_path, manifest, "test", batch_size=2):
        assert x["row_id"].equals(o["row_id"])
        seen.append(item["shard_id"])
    assert seen == ids


def test_footer_close_failure_closes_every_writer_and_handle(tmp_path, monkeypatch):
    from thermoshift import storage

    real_writer = pq.ParquetWriter
    instances = []

    class FailingWriter:
        def __init__(self, sink, *args, **kwargs):
            self.index = len(instances)
            self.sink = sink
            self.real = real_writer(sink, *args, **kwargs)
            self.closed = False
            instances.append(self)

        def write_table(self, *args, **kwargs):
            self.real.write_table(*args, **kwargs)

        def close(self):
            self.real.close()
            self.closed = True
            if self.index == 0:
                raise OSError("injected footer failure")

    initialize(tmp_path, Config(rows=8, episode_steps=8))
    monkeypatch.setattr(storage.pq, "ParquetWriter", FailingWriter)
    with pytest.raises(OSError, match="footer"):
        generate(tmp_path, progress=None)
    assert len(instances) == 2
    assert all(w.closed and w.sink.closed for w in instances)
    assert not list(tmp_path.rglob("*.inprogress"))
    assert not (tmp_path / "_SUCCESS.json").exists()


def test_failure_between_parquet_commit_and_shard_state_is_recoverable(tmp_path, monkeypatch):
    from thermoshift import storage

    real_atomic = storage.atomic_json

    def fail_state(path, value):
        if Path(path).parent.name == "_state":
            raise OSError("injected shard state failure")
        return real_atomic(path, value)

    initialize(tmp_path, Config(rows=73, episode_steps=8))
    monkeypatch.setattr(storage, "atomic_json", fail_state)
    with pytest.raises(OSError, match="shard state"):
        generate(tmp_path, batch_buildings=3, progress=None)
    assert not (tmp_path / "_SUCCESS.json").exists()
    assert not list(tmp_path.rglob("*.inprogress"))
    monkeypatch.setattr(storage, "atomic_json", real_atomic)
    assert generate(tmp_path, batch_buildings=3, progress=None)["written"] == 1
    assert validate(tmp_path, replay=True)["status"] == "passed"


def test_resume_rejects_checkpoint_with_invalid_parquet_schema(release):
    config = load_config(release)
    item = read_json(release / "manifest.json")["files"][0]
    path = release / item["path"]
    table = pq.read_table(path)
    index = table.schema.get_field_index("row_id")
    table = table.set_column(index, pa.field("row_id", pa.int64()), table.column(index))
    pq.write_table(table, path)

    state_path = release / "_state" / f"shard-{item['shard_id']:08d}.json"
    state = read_json(state_path)
    entry = next(file for file in state["files"] if file["path"] == item["path"])
    entry["bytes"], entry["sha256"] = path.stat().st_size, sha256(path)
    atomic_json(state_path, state)

    assert not shard_complete(release, config, item["shard_id"])
    result = generate(release, batch_buildings=7, progress=None)
    assert result["written"] == 1
    assert validate(release)["status"] == "passed"


def test_verified_resume_removes_only_known_shard_temporaries(release):
    item = read_json(release / "manifest.json")["files"][0]
    temporary = (release / item["path"]).with_suffix(".parquet.inprogress")
    temporary.write_bytes(b"abandoned")
    unrelated = release / "unrelated.inprogress"
    unrelated.write_bytes(b"owned by another program")
    generate(release, batch_buildings=7, progress=None)
    assert not temporary.exists() and unrelated.is_file()


def _hold_shared_lease(root, connection):
    with dataset_lock(root, generating=True):
        connection.send("locked")
        connection.recv()


def test_operating_system_releases_lease_when_child_is_terminated(release):
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(target=_hold_shared_lease, args=(str(release), child))
    try:
        process.start()
        child.close()
        assert parent.poll(15) and parent.recv() == "locked"
        with pytest.raises(ValueError, match="busy"):
            validated_manifest(release)
        process.terminate()
        process.join(15)
        assert not process.is_alive()
        validated_manifest(release)
    finally:
        if process.is_alive():
            process.terminate()
            process.join(15)
        parent.close()
        child.close()


@pytest.mark.parametrize(
    "source,error,status",
    [
        ("print('before failure'); raise ValueError('injected')", ValueError, "failed"),
        ("if", SyntaxError, "failed"),
        ("raise KeyboardInterrupt('injected')", KeyboardInterrupt, "interrupted"),
    ],
)
def test_failed_notebook_rerun_clears_stale_success_and_unexecuted_outputs(
    tmp_path, source, error, status
):
    from thermoshift import notebook as runner

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell("print('current'); 2+2"),
            nbformat.v4.new_code_cell(source),
            nbformat.v4.new_code_cell("print('must not execute')"),
            nbformat.v4.new_markdown_cell("Trailing markdown"),
        ]
    )
    notebook.metadata["execution_validation"] = {
        "status": "all_plain_python_cells_executed_top_to_bottom"
    }
    for cell in notebook.cells[:3]:
        cell.execution_count = 99
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="stale output")]
    target = tmp_path / "run.ipynb"
    nbformat.write(notebook, target)
    original = Path.cwd()
    with pytest.raises(error):
        runner.execute(target, tmp_path)
    assert Path.cwd() == original
    saved = nbformat.read(target, as_version=4)
    assert saved.metadata["execution_validation"]["status"] == status
    assert saved.cells[0].execution_count == 1 and saved.cells[1].execution_count == 2
    assert saved.cells[1].outputs[-1].output_type == "error"
    assert saved.cells[2].execution_count is None and saved.cells[2].outputs == []
    assert "stale output" not in target.read_text(encoding="utf-8")
    if error is ValueError:
        assert saved.cells[1].outputs[0].text == "before failure\n"


@pytest.mark.parametrize("operation", ["load", "manifest", "shard", "lease"])
def test_remaining_boolean_api_arguments_are_not_truthiness_checks(release, operation):
    with pytest.raises(ValueError, match="must be boolean"):
        if operation == "load":
            load_config(release, check_runtime="false")
        elif operation == "manifest":
            validated_manifest(release, verify_hash=0)
        elif operation == "shard":
            shard_complete(
                release, Config(rows=1501, episode_steps=24, shard_buildings=17), 0, verify_hash=1
            )
        else:
            with dataset_lock(release, generating="false"):
                pass


def test_initialization_type_error_precedes_filesystem_mutation(tmp_path):
    root = tmp_path / "new-directory"
    with pytest.raises(TypeError, match="Config"):
        initialize(root, {})
    assert not root.exists()


@pytest.mark.parametrize("directory", ["data", "partition", "_state"])
def test_generation_never_writes_through_symlinked_directories(release, directory):
    item = read_json(release / "manifest.json")["files"][0]
    linked = (release / item["path"]).parent if directory == "partition" else release / directory
    moved = release / "redirected-directory"
    linked.rename(moved)
    linked.symlink_to(moved, target_is_directory=True)
    before = {p.relative_to(moved): p.read_bytes() for p in moved.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="symlink"):
        generate(release)
    after = {p.relative_to(moved): p.read_bytes() for p in moved.rglob("*") if p.is_file()}
    assert after == before


def test_generation_never_overwrites_a_symlinked_temporary(release):
    item = read_json(release / "manifest.json")["files"][0]
    path = release / item["path"]
    path.write_bytes(b"corrupt shard forces a rebuild")
    external = release / "unrelated-file"
    external.write_bytes(b"preserve this file")
    temporary = path.with_suffix(".parquet.inprogress")
    temporary.symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        generate(release)
    assert external.read_bytes() == b"preserve this file"
    assert temporary.is_symlink()


@pytest.mark.parametrize("relative", [".lifecycle.lock", "_state/shard-00000000.lock"])
def test_generation_rejects_symlinked_locks(release, relative):
    external = release / "unrelated-lock"
    external.write_bytes(b"preserve this lock")
    lock = release / relative
    lock.unlink(missing_ok=True)
    lock.symlink_to(external)
    with pytest.raises(ValueError, match="symlink"):
        generate(release)
    assert external.read_bytes() == b"preserve this lock"


@pytest.mark.parametrize("failure", ["extra_file", "symlink", "incomplete_shard", "card_write"])
def test_failed_refinalization_clears_old_completion_and_validation(release, monkeypatch, failure):
    from thermoshift import storage

    validate(release)
    if failure == "extra_file":
        (release / "data/extra.parquet").write_bytes(b"unexpected")
    elif failure == "symlink":
        (release / "data/redirect").symlink_to(release / "_state", target_is_directory=True)
    elif failure == "incomplete_shard":
        (release / "_state/shard-00000000.json").unlink()
    else:

        def fail_card(*args, **kwargs):
            raise ValueError("injected card write failure")

        monkeypatch.setattr(storage, "write_card", fail_card)
    with pytest.raises(ValueError):
        finalize(release)
    assert not (release / "_SUCCESS.json").exists()
    assert not (release / "validation.json").exists()


@pytest.mark.parametrize(
    "document,field,value",
    [
        ("manifest.json", None, []),
        ("_SUCCESS.json", None, []),
        ("manifest.json", "fingerprint", None),
        ("manifest.json", "files", None),
        ("manifest.json", "split_rows", []),
        ("manifest.json", "split_rows", {}),
        ("manifest.json", "file", None),
        ("manifest.json", "file.bytes", "100"),
        ("manifest.json", "file.kind", []),
        ("manifest.json", "file.sha256", None),
    ],
)
def test_malformed_release_metadata_raises_value_error(release, document, field, value):
    path = release / document
    content = read_json(path)
    if field is None:
        content = value
    elif field == "file":
        content["files"][0] = value
    elif field.startswith("file."):
        content["files"][0][field.split(".")[1]] = value
    else:
        content[field] = value
    atomic_json(path, content)
    if document == "manifest.json":
        marker = read_json(release / "_SUCCESS.json")
        marker["manifest_sha256"] = sha256(path)
        atomic_json(release / "_SUCCESS.json", marker)
    with pytest.raises(ValueError):
        validated_manifest(release, verify_hash=False)


@pytest.mark.parametrize("name", ["row_id", "building_id", "step"])
def test_public_paired_reader_rejects_mismatched_identifiers(release, name):
    item = next(f for f in read_json(release / "manifest.json")["files"] if f["kind"] == "oracle")
    path = release / item["path"]
    table = pq.ParquetFile(path).read()
    values = table[name].to_numpy().copy()
    values[0] += 1
    index = table.schema.get_field_index(name)
    table = table.set_column(index, table.schema.field(index), pa.array(values))
    pq.write_table(table, path)
    resign(release)
    with pytest.raises(ValueError, match="join mismatch"):
        list(iter_pairs(release, batch_size=13))


def test_schema_repair_flag_requires_a_boolean(tmp_path):
    with pytest.raises(ValueError, match="repair_missing must be boolean"):
        check_schema_documents(tmp_path, repair_missing="false")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field", ["shard_id", "file.shard_id", "file.rows", "file.sha256"])
def test_malformed_shard_state_is_never_resumed(release, field):
    path = release / "_state/shard-00000000.json"
    state = read_json(path)
    if field == "shard_id":
        state[field] = False
    elif field == "file.shard_id":
        state["files"][0]["shard_id"] = False
    elif field == "file.rows":
        state["files"][0]["rows"] = float(state["files"][0]["rows"])
    else:
        del state["files"][0]["sha256"]
    atomic_json(path, state)
    assert not shard_complete(release, load_config(release), 0, verify_hash=False)
    assert generate(release)["written"] == 1
    assert validate(release, replay=True)["status"] == "passed"
