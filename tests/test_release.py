# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from thermoshift import Config
from thermoshift.filesystem import atomic_json, read_json, sha256
from thermoshift.provenance import load_config
from thermoshift.shards import shard_path
from thermoshift.storage import finalize, generate, initialize
from thermoshift.validation import validate


def canonical(root, kind):
    tables = [
        pq.ParquetFile(path).read()
        for path in sorted((Path(root) / "data" / kind).rglob("*.parquet"))
    ]
    table = pa.concat_tables(tables)
    return table.sort_by("row_id")


@pytest.mark.parametrize(
    "rows,steps,confounded", [(1, 168, False), (16003, 168, False), (1100, 24, True)]
)
def test_exact_counts_and_scientific_invariants(tmp_path, rows, steps, confounded):
    config = Config(
        rows=rows, episode_steps=steps, hidden_confounding=confounded, shard_buildings=31
    )
    initialize(tmp_path, config)
    generate(tmp_path, batch_buildings=17, progress=None)
    report = validate(tmp_path, batch_size=53)
    assert report["rows"] == rows
    assert report["status"] == "passed"
    assert canonical(tmp_path, "logged").num_rows == rows
    assert canonical(tmp_path, "oracle").num_rows == rows


def test_workers_shards_and_batches_preserve_records(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    initialize(a, Config(rows=3007, episode_steps=24, shard_buildings=32))
    initialize(b, Config(rows=3007, episode_steps=24, shard_buildings=19))
    generate(a, workers=1, batch_buildings=13, progress=None)
    generate(b, workers=2, batch_buildings=7, progress=None)
    for kind in ("logged", "oracle"):
        assert canonical(a, kind).equals(canonical(b, kind))
    validate(b, batch_size=31)


def test_checksum_resume_and_same_size_corruption_recovery(tmp_path):
    initialize(tmp_path, Config(rows=3017, episode_steps=24, shard_buildings=32))
    first = generate(tmp_path, batch_buildings=15, progress=None)
    files = read_json(tmp_path / "manifest.json")["files"]
    hashes = {f["path"]: sha256(tmp_path / f["path"]) for f in files}
    resumed = generate(tmp_path, batch_buildings=15, progress=None)
    assert resumed["resumed"] == first["written"]
    path = tmp_path / files[0]["path"]
    with path.open("r+b") as handle:
        handle.seek(11)
        byte = handle.read(1)
        handle.seek(11)
        handle.write(bytes([byte[0] ^ 255]))
    with pytest.raises(ValueError, match="checksum"):
        validate(tmp_path)
    repaired = generate(tmp_path, batch_buildings=15, progress=None)
    assert repaired["written"] == 1
    assert {f: sha256(tmp_path / f) for f in hashes} == hashes
    validate(tmp_path)


def test_generation_clears_the_completion_marker_and_validation_report(tmp_path):
    # A release being written must not still advertise itself as complete and
    # validated. A single-rank run finalizes at the end and rewrites both files, so
    # the distributed path is where this matters: it never finalizes, and stale
    # documents would survive a rerun describing partly regenerated data.
    initialize(tmp_path, Config(rows=200, episode_steps=24, shard_buildings=5))
    generate(tmp_path, progress=None)
    validate(tmp_path)
    assert (tmp_path / "_SUCCESS.json").is_file()
    assert (tmp_path / "validation.json").is_file()

    generate(tmp_path, rank=0, world_size=2, progress=None)
    assert not (tmp_path / "_SUCCESS.json").exists()
    assert not (tmp_path / "validation.json").exists()

    generate(tmp_path, rank=1, world_size=2, progress=None)
    finalize(tmp_path)
    validate(tmp_path)


def test_distributed_rank_coverage_and_incomplete_finalize(tmp_path):
    initialize(tmp_path, Config(rows=3001, episode_steps=24, shard_buildings=24))
    generate(tmp_path, rank=0, world_size=2, batch_buildings=8, progress=None)
    with pytest.raises(ValueError, match="incomplete"):
        finalize(tmp_path)
    generate(tmp_path, rank=1, world_size=2, batch_buildings=8, progress=None)
    manifest = finalize(tmp_path)
    assert manifest["rows"] == 3001
    validate(tmp_path, batch_size=41)


def test_config_change_and_unrelated_files_are_rejected(tmp_path):
    root = tmp_path / "release"
    initialize(root, Config(rows=100))
    with pytest.raises(ValueError, match="different configuration"):
        initialize(root, Config(rows=200))
    generate(root, progress=None)
    extra = root / "data" / "extra.parquet"
    extra.write_bytes(b"not a real parquet file")
    with pytest.raises(ValueError, match="unexpected"):
        finalize(root)


def test_runtime_mismatch_prevents_mixed_resume(tmp_path):
    initialize(tmp_path, Config(rows=10))
    document = read_json(tmp_path / "run_config.json")
    document["runtime"]["numpy"] = "different-version"
    atomic_json(tmp_path / "run_config.json", document)
    with pytest.raises(ValueError, match="runtime differs"):
        generate(tmp_path, progress=None)


def test_package_rename_preserves_legacy_provenance_reading(tmp_path):
    config = Config(rows=10)
    initialize(tmp_path, config)
    document = read_json(tmp_path / "run_config.json")
    assert "thermoshift" in document["runtime"]
    assert "thermoshift-synth" not in document["runtime"]
    document["runtime"]["thermoshift-synth"] = document["runtime"].pop("thermoshift")
    atomic_json(tmp_path / "run_config.json", document)
    assert load_config(tmp_path) == config
    with pytest.raises(ValueError, match="runtime differs"):
        generate(tmp_path)


def test_runtime_requires_package_version_after_rename(tmp_path):
    initialize(tmp_path, Config(rows=10))
    document = read_json(tmp_path / "run_config.json")
    del document["runtime"]["thermoshift"]
    atomic_json(tmp_path / "run_config.json", document)
    with pytest.raises(ValueError, match="invalid runtime provenance"):
        load_config(tmp_path)


def test_hugging_face_local_loading_and_column_projection(tmp_path):
    from datasets import load_dataset

    initialize(tmp_path, Config(rows=16803, shard_buildings=33))
    generate(tmp_path, progress=None)
    card = (tmp_path / "README.md").read_text()
    metadata = yaml.safe_load(card.split("---")[1])
    assert [c["config_name"] for c in metadata["configs"]] == ["logged", "oracle"]
    for kind in ("logged", "oracle"):
        for split in ("train", "validation", "test", "test_heatwave", "test_sensor"):
            ds = load_dataset(
                str(tmp_path),
                name=kind,
                split=split,
                streaming=True,
                columns=["row_id", "building_id", "step"],
            )
            count = sum(1 for _ in ds)
            assert count == read_json(tmp_path / "manifest.json")["split_rows"][split]


def test_corrupt_target_detected_even_with_updated_checksum(tmp_path):
    initialize(tmp_path, Config(rows=501, episode_steps=24))
    generate(tmp_path, progress=None)
    manifest = read_json(tmp_path / "manifest.json")
    entry = next(f for f in manifest["files"] if f["kind"] == "logged")
    path = tmp_path / entry["path"]
    table = pq.ParquetFile(path).read()
    column = table["y_cost"].to_numpy().copy()
    column[0] += 10
    table = table.set_column(
        table.schema.get_field_index("y_cost"), table.schema.field("y_cost"), pa.array(column)
    )
    pq.write_table(table, path)
    entry["sha256"], entry["bytes"] = sha256(path), path.stat().st_size
    manifest["bytes"] = sum(f["bytes"] for f in manifest["files"])
    atomic_json(tmp_path / "manifest.json", manifest)
    atomic_json(
        tmp_path / "_SUCCESS.json",
        {
            **read_json(tmp_path / "_SUCCESS.json"),
            "manifest_sha256": sha256(tmp_path / "manifest.json"),
        },
    )
    with pytest.raises(ValueError, match="counterfactual mismatch"):
        validate(tmp_path)


def test_shard_paths_match_the_documented_release_layout():
    # README and docs/generation.md publish this layout as
    # data/<kind>/<split>/<bucket>/part-*.parquet and consumers read the tree
    # directly. Every path in the package comes from this one function, so a changed
    # bucket size or field width stays self-consistent and silently reshapes the
    # published release instead of failing.
    assert shard_path("logged", "train", 0) == "data/logged/train/00000/part-00000000.parquet"
    assert shard_path("oracle", "test", 1) == "data/oracle/test/00000/part-00000001.parquet"
    # A thousand shards per bucket directory, and the widths the names are padded
    # to; the rollover is what fixes the divisor.
    assert shard_path("logged", "train", 999) == "data/logged/train/00000/part-00000999.parquet"
    assert shard_path("logged", "train", 1000) == "data/logged/train/00001/part-00001000.parquet"
    assert (
        shard_path("logged", "test_heatwave", 1_000_000)
        == "data/logged/test_heatwave/01000/part-01000000.parquet"
    )


def test_bundled_sample_release_matches_the_layout_the_generator_produces():
    # sample_dataset/ is a committed release README tells readers to load. It is
    # pruned from the source distribution, so guard for it not being on disk.
    root = Path(__file__).resolve().parents[1] / "sample_dataset"
    if not (root / "manifest.json").is_file():
        pytest.skip("bundled sample release is not present in this checkout")
    files = read_json(root / "manifest.json")["files"]
    assert files, "bundled sample release lists no files"
    for item in files:
        assert item["path"] == shard_path(item["kind"], item["split"], item["shard_id"])
        assert (root / item["path"]).is_file()
