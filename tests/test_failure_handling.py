# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Failure recovery, API boundaries and resource cleanup."""

import multiprocessing
import shutil
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


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("short_digest", "invalid source SHA-256"),
        ("uppercase_digest", "invalid source SHA-256"),
        ("digest_not_a_string", "invalid source SHA-256"),
        ("fingerprint", "fingerprint mismatch"),
        ("config_value", "fingerprint mismatch"),
    ],
)
def test_tampered_provenance_digests_are_rejected(release, mutation, message):
    document = read_json(release / "run_config.json")
    if mutation == "short_digest":
        document["source_sha256"] = "abc123"
    elif mutation == "uppercase_digest":
        document["source_sha256"] = document["source_sha256"].upper()
    elif mutation == "digest_not_a_string":
        document["source_sha256"] = ["0" * 64]
    elif mutation == "fingerprint":
        document["fingerprint"] = "0" * 64
    else:
        # A changed setting no longer hashes to the recorded fingerprint.
        document["config"]["seed"] = document["config"]["seed"] + 1
    atomic_json(release / "run_config.json", document)
    with pytest.raises(ValueError, match=message):
        load_config(release)


@pytest.mark.parametrize(
    "operation", [generate, finalize, validate, validated_manifest, iter_pairs]
)
def test_operations_name_a_missing_dataset_directory(tmp_path, operation):
    missing = tmp_path / "never-initialized"
    with pytest.raises(ValueError, match="does not exist; initialize it first"):
        result = operation(missing)
        if operation is iter_pairs:
            next(result)


def test_shard_state_row_counts_must_match_the_plan(release):
    config = load_config(release)
    path = release / "_state/shard-00000000.json"
    state = read_json(path)
    state["files"][0]["rows"] += 1
    atomic_json(path, state)
    assert not shard_complete(release, config, 0, verify_hash=False)


def test_a_failed_atomic_write_leaves_the_previous_file_and_no_temporary(tmp_path, monkeypatch):
    # "Atomic" means the destination is only ever reached by the rename, so a
    # failure cannot leave it half written, and the temporary never outlives the
    # attempt.
    import os

    from thermoshift.filesystem import atomic_bytes

    target = tmp_path / "nested" / "document.json"
    target.parent.mkdir()
    target.write_bytes(b"original\n")

    def fail(*args, **kwargs):
        raise OSError("injected rename failure")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="injected rename failure"):
        atomic_bytes(target, b"replacement\n")
    assert target.read_bytes() == b"original\n"
    assert [p.name for p in target.parent.iterdir()] == ["document.json"]


def test_atomic_json_refuses_nonfinite_numbers_instead_of_writing_invalid_json(tmp_path):
    # NaN and the infinities are not JSON. Python writes them as bare NaN and
    # Infinity tokens that a strict parser rejects, so a release document holding
    # one would be unreadable outside Python rather than merely wrong.
    from thermoshift.filesystem import atomic_json

    for value in (float("nan"), float("inf"), float("-inf")):
        target = tmp_path / "document.json"
        with pytest.raises(ValueError, match="not JSON compliant"):
            atomic_json(target, {"value": value})
        assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_json_does_not_depend_on_the_order_keys_were_added(tmp_path):
    # The completion marker binds manifest.json by its bytes, so the same content
    # has to serialize the same way however the document was assembled.
    from thermoshift.filesystem import atomic_json

    first, second = tmp_path / "a.json", tmp_path / "b.json"
    atomic_json(first, {"zebra": 1, "alpha": {"n": 2, "m": 3}, "middle": 4})
    atomic_json(second, {"middle": 4, "alpha": {"m": 3, "n": 2}, "zebra": 1})
    assert first.read_bytes() == second.read_bytes()


def test_commit_refuses_a_destination_redirected_after_the_temporary_opened(tmp_path, monkeypatch):
    # The destination is resolved again at commit time because nothing keeps a
    # path symlink-free between opening the temporary and replacing it into
    # place. Joining the pieces directly instead would follow a parent that
    # turned into a redirect and land the shard outside the release entirely.
    from thermoshift import storage

    initialize(tmp_path, Config(rows=200, episode_steps=24))
    real_expected = storage.expected_shard
    redirected = []

    def redirect_then_delegate(config, shard_id):
        # Called once the writers have produced their temporaries and just before
        # the commit loop resolves each destination. Point the redirect at the
        # real directory so every temporary stays reachable and the only thing
        # that has changed is that a component is now a symlink.
        if not redirected:
            for parent in sorted((tmp_path / "data/logged").iterdir()):
                if parent.is_dir() and not parent.is_symlink():
                    moved = parent.with_name(parent.name + "-real")
                    parent.rename(moved)
                    parent.symlink_to(moved, target_is_directory=True)
                    redirected.append(parent)
                    break
        return real_expected(config, shard_id)

    monkeypatch.setattr(storage, "expected_shard", redirect_then_delegate)
    with pytest.raises(ValueError, match="symlinks and junctions are not allowed"):
        generate(tmp_path, progress=None)
    assert redirected, "the redirect was never installed, so nothing was tested"
    assert not (tmp_path / "_SUCCESS.json").exists()


def reseal(root):
    """Re-sign the completion marker only, leaving the manifest exactly as edited."""
    atomic_json(
        root / "_SUCCESS.json",
        {
            "manifest_sha256": sha256(root / "manifest.json"),
            "metadata_sha256": metadata_hashes(root),
        },
    )


@pytest.mark.parametrize(
    "tamper,message",
    [
        ("manifest_rows", "total row count mismatch"),
        ("manifest_bytes", "manifest byte accounting mismatch"),
        ("manifest_buildings", "total building count mismatch"),
        ("file_bytes", "file size mismatch"),
        ("file_rows", "manifest does not cover the configured rows"),
        ("file_sha256_format", "invalid file SHA-256"),
        ("file_shard_id", "shard range mismatch"),
        ("split_rows", "split row count mismatch"),
        ("parquet_rows", "Parquet row count mismatch"),
        ("parquet_schema", "Arrow schema mismatch"),
    ],
)
def test_wellformed_but_wrong_release_metadata_is_rejected_without_hashes(release, tamper, message):
    # Every type stays valid and only the value is wrong, which is what a tampered
    # or half-rewritten release looks like. Run with verify_hash disabled: that
    # documented fast path skips the per-file checksums, leaving these structural
    # checks as the only thing standing behind it.
    manifest = read_json(release / "manifest.json")
    entry = manifest["files"][0]
    target = release / entry["path"]
    if tamper == "manifest_rows":
        manifest["rows"] += 1
    elif tamper == "manifest_bytes":
        manifest["bytes"] += 1
    elif tamper == "manifest_buildings":
        manifest["buildings"] += 1
    elif tamper == "file_sha256_format":
        entry["sha256"] = "z" * 64
    elif tamper == "file_shard_id":
        entry["shard_id"] += 10_000
    elif tamper == "file_rows":
        entry["rows"] += 1
    elif tamper == "split_rows":
        manifest["split_rows"]["train"] += 1
    elif tamper == "file_bytes":
        # Keep the aggregate consistent so the per-file size check is what fires.
        entry["bytes"] += 1
        manifest["bytes"] = sum(f["bytes"] for f in manifest["files"])
    else:
        table = pq.ParquetFile(target).read()
        if tamper == "parquet_rows":
            table = table.slice(0, table.num_rows - 1)
        else:
            table = table.append_column("extra", pa.array([0] * table.num_rows, type=pa.int8()))
        pq.write_table(table, target)
        # Make the release agree with the rewritten file everywhere else, so only
        # the Parquet footer can tell that its contents are wrong.
        entry["bytes"], entry["sha256"] = target.stat().st_size, sha256(target)
        manifest["bytes"] = sum(f["bytes"] for f in manifest["files"])
    atomic_json(release / "manifest.json", manifest)
    reseal(release)
    with pytest.raises(ValueError, match=message):
        validated_manifest(release, verify_hash=False)


@pytest.mark.parametrize("change", ["stray", "missing"])
def test_data_tree_must_hold_exactly_the_files_the_manifest_lists(release, change):
    # finalize() refuses to seal a release with unexpected files under data/, and
    # the reader has to make the same judgement about one it did not build: a file
    # added or removed afterwards leaves every manifest entry internally consistent,
    # so comparing the tree against the manifest is the only check that looks at
    # what is on disk.
    entry = read_json(release / "manifest.json")["files"][0]
    listed = release / entry["path"]
    if change == "stray":
        shutil.copyfile(listed, listed.parent / "part-00009999.parquet")
    else:
        listed.unlink()
    with pytest.raises(ValueError, match="unexpected/missing Parquet file"):
        validated_manifest(release, verify_hash=False)


def test_skipping_hashes_still_reads_a_release_whose_checksums_are_wrong(release):
    # The documented meaning of verify_hash=False: a checksum that is well formed
    # but wrong is simply not consulted. Asserting it keeps the fast path honest
    # about what it does and does not promise.
    manifest = read_json(release / "manifest.json")
    manifest["files"][0]["sha256"] = "0" * 64
    atomic_json(release / "manifest.json", manifest)
    reseal(release)
    assert validated_manifest(release, verify_hash=False)
    with pytest.raises(ValueError, match="checksum mismatch"):
        validated_manifest(release, verify_hash=True)


def test_manifest_edited_after_finalize_is_rejected_by_the_completion_marker(release):
    # Every manifest field is cross-checked against the configuration or the files,
    # so the content checks pin all of them. What they cannot see is an added key,
    # which a reader would then be trusting although finalize never wrote it.
    # Binding the manifest bytes to the completion marker is what catches that.
    path = release / "manifest.json"
    manifest = read_json(path)
    assert validated_manifest(release)  # the untouched release validates
    manifest["retracted"] = True
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="manifest is not finalized"):
        validated_manifest(release)
    # Re-signing the marker makes it a legitimately finalized release again, so
    # the marker is doing this on its own rather than some later content check.
    resign(release)
    assert validated_manifest(release)


def _shard_state(release, shard_id=0):
    return release / f"_state/shard-{shard_id:08d}.json"


# A checkpoint that does not belong to this release would let resume mix shards
# from different configurations into one output. initialize() and generate()
# reject a changed configuration or runtime before resume gets this far, so these
# cover the checkpoint's own contract: a state file copied in from elsewhere, or
# a shard that rotted after it was committed.
def test_shard_state_from_a_different_configuration_is_not_reused(release):
    config = load_config(release)
    path = _shard_state(release)
    state = read_json(path)
    assert shard_complete(release, config, 0)
    state["fingerprint"] = "0" * 64
    atomic_json(path, state)
    assert not shard_complete(release, config, 0)


def test_shard_state_is_not_reused_when_its_provenance_digest_disagrees(release):
    config = load_config(release)
    path = _shard_state(release)
    state = read_json(path)
    state["provenance_sha256"] = "0" * 64
    atomic_json(path, state)
    assert not shard_complete(release, config, 0)


def test_self_consistent_shard_state_with_wrong_parquet_contents_is_rejected(release):
    # The footer check exists for exactly this: a checkpoint that agrees with
    # itself on size and checksum, and with the plan on row counts, while the
    # Parquet file behind it holds a different number of rows. Every other guard
    # passes, so without the footer this shard would be kept and finalized.
    config = load_config(release)
    path = _shard_state(release)
    state = read_json(path)
    entry = state["files"][0]
    target = release / entry["path"]
    table = pq.ParquetFile(target).read()
    pq.write_table(table.slice(0, table.num_rows - 1), target)
    entry["bytes"], entry["sha256"] = target.stat().st_size, sha256(target)
    atomic_json(path, state)
    assert entry["rows"] == read_json(path)["files"][0]["rows"]  # still the planned count
    assert not shard_complete(release, config, 0)


def test_shard_state_agreeing_with_its_file_but_not_the_plan_is_rejected(release):
    # The mirror of the case above. Here the checkpoint and its Parquet file
    # agree with each other and only the plan disagrees, which is what a shard
    # left behind by a different row budget looks like. The footer check cannot
    # see it, so the comparison against the planned counts is what rejects it.
    config = load_config(release)
    path = _shard_state(release)
    state = read_json(path)
    entry = state["files"][0]
    target = release / entry["path"]
    table = pq.ParquetFile(target).read()
    pq.write_table(table.slice(0, table.num_rows - 1), target)
    entry["rows"] = table.num_rows - 1
    entry["bytes"], entry["sha256"] = target.stat().st_size, sha256(target)
    atomic_json(path, state)
    assert not shard_complete(release, config, 0)


def test_shard_file_replaced_by_a_symlink_is_not_reused(release):
    config = load_config(release)
    entry = read_json(_shard_state(release))["files"][0]
    target = release / entry["path"]
    moved = target.with_suffix(".moved")
    target.rename(moved)
    target.symlink_to(moved)
    # The bytes behind the link are the committed ones, so size and checksum both
    # still agree; only refusing the redirect rejects it.
    assert sha256(target) == entry["sha256"]
    assert not shard_complete(release, config, 0)


@pytest.mark.parametrize("damage", ["truncated", "emptied"])
def test_shard_file_whose_size_no_longer_matches_is_not_reused(release, damage):
    config = load_config(release)
    entry = read_json(_shard_state(release))["files"][0]
    target = release / entry["path"]
    keep = 0 if damage == "emptied" else entry["bytes"] // 2
    with target.open("r+b") as handle:
        handle.truncate(keep)
    assert target.stat().st_size != entry["bytes"]
    # Size alone has to reject these: an empty or half-written shard has no
    # readable footer, so a later guard would raise rather than return False.
    assert not shard_complete(release, config, 0, verify_hash=False)


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


def test_worker_death_is_reported_with_a_remedy(tmp_path, monkeypatch):
    # A worker that raises reports through its future; one killed by the system
    # only breaks the pool, which must not surface as an unhandled RuntimeError.
    from concurrent.futures.process import BrokenProcessPool

    from thermoshift import storage

    class DeadPool:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *details):
            return False

        def submit(self, *args, **kwargs):
            raise BrokenProcessPool("mock worker terminated abruptly")

    initialize(tmp_path, Config(rows=96, episode_steps=8, shard_buildings=2))
    monkeypatch.setattr(storage, "ProcessPoolExecutor", DeadPool)
    with pytest.raises(OSError, match="worker exited without reporting") as failure:
        generate(tmp_path, workers=2, progress=None)
    assert isinstance(failure.value.__cause__, BrokenProcessPool)
    assert not (tmp_path / "_SUCCESS.json").exists()


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


@pytest.mark.parametrize(
    "relative",
    ["/etc/passwd", "../escape.json", "data/../../escape.json", "data/logged/../../.."],
)
def test_dataset_paths_cannot_escape_the_release_directory(tmp_path, relative):
    # "/etc/passwd" is the cross-platform case: Windows reports it as not
    # absolute because it carries no drive, yet joining it resets to the drive
    # root, so a guard written on is_absolute() alone lets it out of the release.
    from thermoshift.filesystem import dataset_path

    with pytest.raises(ValueError, match="must be relative to the dataset directory"):
        dataset_path(tmp_path, relative)


def test_junctions_are_refused_like_symlinks(tmp_path, monkeypatch):
    # A Windows junction redirects like a symlink but is_symlink() does not
    # report one, so the guard asks os.path.isjunction as well. That returns
    # False off Windows, so drive the detector directly to prove the wiring:
    # no CI platform here can create a junction.
    from thermoshift import filesystem

    target = tmp_path / "data" / "logged"
    target.mkdir(parents=True)
    assert filesystem.dataset_path(tmp_path, "data/logged").is_dir()

    monkeypatch.setattr(filesystem, "_isjunction", lambda path: Path(path) == target)
    with pytest.raises(ValueError, match="symlinks and junctions are not allowed"):
        filesystem.dataset_path(tmp_path, "data/logged/train/part-0.parquet")
    # Siblings of the junction stay usable.
    assert filesystem.dataset_path(tmp_path, "data/oracle").name == "oracle"


def test_initialization_removes_a_stale_config_temporary_but_not_foreign_files(tmp_path):
    # A process can die between mkstemp and the atomic rename of run_config.json.
    stale = tmp_path / "run_config.json.tmp-abcdef"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"partial write")
    initialize(tmp_path, Config(rows=24, episode_steps=8))
    assert not stale.exists() and (tmp_path / "run_config.json").is_file()

    other = tmp_path.parent / "second"
    other.mkdir()
    (other / "unrelated.txt").write_text("someone else owns this directory")
    with pytest.raises(ValueError, match="must be empty"):
        initialize(other, Config(rows=24, episode_steps=8))
    assert (other / "unrelated.txt").is_file()
    assert not (other / "run_config.json").exists()


@pytest.mark.parametrize(
    "repo_id,revision,message",
    [
        # The SDK accepts a bare name; a release still needs its namespace.
        ("justaname", "main", "OWNER/DATASET"),
        ("owner/set", "", "existing branch"),
        ("owner/set", "   ", "existing branch"),
        ("owner/set", None, "existing branch"),
    ],
)
def test_publication_targets_are_checked_before_any_validation(
    tmp_path, monkeypatch, repo_id, revision, message
):
    from thermoshift import validation

    def forbidden(*args, **kwargs):
        raise AssertionError("validation should not run for an invalid publication target")

    monkeypatch.setattr(validation, "_validate_locked", forbidden)
    with pytest.raises(ValueError, match=message):
        publish(tmp_path / "does-not-exist", repo_id, revision=revision)


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


@pytest.mark.parametrize(
    "failure", ["extra_file", "symlink", "state_symlink", "incomplete_shard", "card_write"]
)
def test_failed_refinalization_clears_old_completion_and_validation(release, monkeypatch, failure):
    from thermoshift import storage

    validate(release)
    if failure == "extra_file":
        (release / "data/extra.parquet").write_bytes(b"unexpected")
    elif failure == "symlink":
        (release / "data/redirect").symlink_to(release / "_state", target_is_directory=True)
    elif failure == "state_symlink":
        # A redirected shard record must be named as such, not reported as an
        # unrelated incomplete shard.
        record = release / "_state/shard-00000000.json"
        moved = release / "redirected-state.json"
        record.rename(moved)
        record.symlink_to(moved)
    elif failure == "incomplete_shard":
        (release / "_state/shard-00000000.json").unlink()
    else:

        def fail_card(*args, **kwargs):
            raise ValueError("injected card write failure")

        monkeypatch.setattr(storage, "write_card", fail_card)
    expected = "symlinks are not allowed under _state/" if failure == "state_symlink" else None
    with pytest.raises(ValueError, match=expected):
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


@pytest.mark.parametrize(
    "field",
    ["shard_id", "file.shard_id", "file.rows", "file.sha256", "files_not_a_list", "files_short"],
)
def test_malformed_shard_state_is_never_resumed(release, field):
    path = release / "_state/shard-00000000.json"
    state = read_json(path)
    if field == "shard_id":
        state[field] = False
    elif field == "file.shard_id":
        state["files"][0]["shard_id"] = False
    elif field == "file.rows":
        state["files"][0]["rows"] = float(state["files"][0]["rows"])
    elif field == "files_not_a_list":
        state["files"] = {f["path"]: f for f in state["files"]}
    elif field == "files_short":
        state["files"] = state["files"][:-1]
    else:
        del state["files"][0]["sha256"]
    atomic_json(path, state)
    assert not shard_complete(release, load_config(release), 0, verify_hash=False)
    assert generate(release)["written"] == 1
    assert validate(release, replay=True)["status"] == "passed"
