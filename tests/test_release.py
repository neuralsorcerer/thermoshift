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
