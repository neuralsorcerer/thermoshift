# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Physical equations, data integrity and boundary conditions."""

from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.integrate import solve_ivp

from thermoshift import Config
from thermoshift.filesystem import atomic_json, read_json, sha256
from thermoshift.lifecycle import dataset_lock
from thermoshift.provenance import metadata_hashes, source_digest
from thermoshift.reading import iter_pairs, validated_manifest
from thermoshift.simulator import simulate, thermal_step
from thermoshift.storage import finalize, generate, initialize
from thermoshift.validation import validate


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


@pytest.fixture
def release(tmp_path):
    initialize(tmp_path, Config(rows=1501, episode_steps=24, shard_buildings=17))
    generate(tmp_path, batch_buildings=7, progress=None)
    return tmp_path


def test_rc_equation_against_independent_adaptive_numerical_ode():
    rng = np.random.default_rng(936)
    for _ in range(40):
        temp, outdoor = rng.uniform(10, 50, 2)
        g, c = rng.uniform(0.05, 2), rng.uniform(1, 20)
        heat, cooling = rng.uniform(-1, 12), rng.uniform(0, 30)
        solution = solve_ivp(
            lambda _, y, g=g, outdoor=outdoor, heat=heat, cooling=cooling, c=c: (
                (g * (outdoor - y) + heat - cooling) / c
            ),
            (0, 1),
            [temp],
            rtol=1e-11,
            atol=1e-12,
        )
        assert solution.success
        assert thermal_step(temp, outdoor, g, c, heat, cooling) == pytest.approx(
            solution.y[0, -1], abs=1e-9
        )


def test_longest_episode_weekdays_and_boundary():
    x, _, _ = simulate(Config(rows=8760, episode_steps=8760), 0, 1)
    days = x["day_of_week"].to_numpy().astype(int)
    assert np.array_equal(days, (days[0] + np.arange(8760) // 24) % 7)
    assert x["episode_end"].to_numpy().sum() == 1


def test_maximum_int64_row_identifiers_do_not_overflow():
    c = Config(rows=2**63 - 1, episode_steps=24)
    x, o, _ = simulate(c, c.buildings - 1, c.buildings)
    ids = x["row_id"].to_numpy()
    assert ids[-1] == c.rows - 1 and np.all(np.diff(ids) == 1)
    assert x["row_id"].equals(o["row_id"])
    assert x["episode_end"][-1].as_py()


def test_sensor_fault_uses_serialized_threshold(monkeypatch):
    import thermoshift.simulator as module

    original = module.normal

    def draw(ids, seed, stream, step=0):
        if stream == 12:
            return np.full(len(ids), (0.75 + 1e-8) / 0.12)
        if stream == 13:
            return np.zeros(len(ids))
        return original(ids, seed, stream, step)

    monkeypatch.setattr(module, "normal", draw)
    x, o, _ = simulate(Config(rows=48, episode_steps=24), 0, 2)
    expected = np.isnan(x["obs_temp_c"].to_numpy()) | (np.abs(o["sensor_bias_c"].to_numpy()) > 0.75)
    assert np.array_equal(o["sensor_fault"].to_numpy(), expected)


@pytest.mark.parametrize("field", ["exploration", "comfort_weight", "carbon_weight"])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0.2", None])
def test_bad_numeric_config_inputs(field, value):
    with pytest.raises(ValueError):
        replace(Config(), **{field: value})


def test_missing_initialization_metadata_is_repaired_but_corruption_is_rejected(tmp_path):
    c = Config(rows=20)
    initialize(tmp_path, c)
    (tmp_path / "schema.json").unlink()
    initialize(tmp_path, c)
    assert (tmp_path / "schema.json").is_file()
    atomic_json(tmp_path / "feature_roles.json", {})
    with pytest.raises(ValueError, match="feature_roles"):
        initialize(tmp_path, c)


@pytest.mark.parametrize(
    "name", ["schema.json", "feature_roles.json", "README.md", "DATASHEET.md", "LICENSE"]
)
def test_missing_release_metadata_fails(release, name):
    (release / name).unlink()
    with pytest.raises((ValueError, OSError)):
        validate(release)


def test_modified_card_checksum_fails(release):
    (release / "README.md").write_text("incorrect release information")
    with pytest.raises(ValueError, match="metadata checksum"):
        validated_manifest(release)


def test_finalize_checks_corrupted_shard_bytes(release):
    item = read_json(release / "manifest.json")["files"][0]
    path = release / item["path"]
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="incomplete"):
        finalize(release)
    generate(release, batch_buildings=11, progress=None)
    assert validate(release, batch_size=7)["status"] == "passed"


def test_manifest_order_and_building_count_are_checked(release):
    manifest = read_json(release / "manifest.json")
    manifest["files"].reverse()
    atomic_json(release / "manifest.json", manifest)
    resign(release)
    with pytest.raises(ValueError, match="canonical"):
        validated_manifest(release)
    manifest["files"].reverse()
    manifest["buildings"] += 1
    atomic_json(release / "manifest.json", manifest)
    resign(release)
    with pytest.raises(ValueError, match="building count"):
        validated_manifest(release)


def test_lifecycle_excludes_mutations_and_readers_but_allows_ranks(release):
    with dataset_lock(release, generating=True):
        with dataset_lock(release, generating=True):
            pass
        for operation in (finalize, validate, validated_manifest):
            with pytest.raises(ValueError, match="busy"):
                operation(release)
    with dataset_lock(release), pytest.raises(ValueError, match="busy"):
        generate(release, progress=None)


def test_iter_pairs_holds_snapshot_until_closed(release):
    pairs = iter_pairs(release, batch_size=17)
    next(pairs)
    with pytest.raises(ValueError, match="busy"):
        generate(release, progress=None)
    pairs.close()
    validated_manifest(release)


def test_sensor_history_corruption_fails_even_after_checksums_are_updated(release):
    for item in read_json(release / "manifest.json")["files"]:
        if item["kind"] != "logged":
            continue
        path = release / item["path"]
        table = pq.read_table(path)
        missing = table["obs_temp_c"].is_null().to_numpy()
        candidates = np.flatnonzero(missing & (table["step"].to_numpy() > 0))
        if len(candidates):
            data = table["obs_temp_last_c"].to_numpy().copy()
            data[candidates[0]] += 0.25
            i = table.schema.get_field_index("obs_temp_last_c")
            table = table.set_column(i, table.schema.field(i), pa.array(data))
            pq.write_table(table, path)
            resign(release)
            with pytest.raises(ValueError, match="sensor history"):
                validate(release, batch_size=7)
            return
    pytest.fail("deterministic fixture must contain a sensor dropout")


@pytest.mark.parametrize(
    "rows,steps,hidden,exploration,weight",
    [
        (1, 2, False, 1.0, 0.0),
        (77, 24, True, 0.15, 999999.0),
        (2501, 24, False, 0.000001, 0.3),
        (1001, 24, False, 1.0, 0.3),
    ],
)
def test_configuration_edges_and_small_validation_batches(
    tmp_path, rows, steps, hidden, exploration, weight
):
    initialize(
        tmp_path,
        Config(
            rows=rows,
            episode_steps=steps,
            hidden_confounding=hidden,
            exploration=exploration,
            comfort_weight=weight,
            carbon_weight=weight,
        ),
    )
    generate(tmp_path, progress=None)
    assert validate(tmp_path, batch_size=13)["status"] == "passed"


def test_replay_and_argument_checks(tmp_path):
    initialize(tmp_path, Config(rows=91, episode_steps=8))
    generate(tmp_path, batch_buildings=3, progress=None)
    assert validate(tmp_path, batch_size=5, replay=True)["replay"] is True
    for kwargs in ({"batch_size": 0}, {"full": False, "replay": True}):
        with pytest.raises(ValueError):
            validate(tmp_path, **kwargs)
    for kwargs in ({"rank": True}, {"workers": 0}, {"world_size": 1.2}, {"batch_buildings": -1}):
        with pytest.raises(ValueError):
            generate(tmp_path, **kwargs)
    assert len(source_digest()) == 64


def test_negative_age_on_available_reading_is_rejected(release):
    item = next(f for f in read_json(release / "manifest.json")["files"] if f["kind"] == "logged")
    path = release / item["path"]
    table = pq.read_table(path)
    index = np.flatnonzero(~table["obs_temp_c"].is_null().to_numpy())[0]
    age = table["sensor_age_steps"].to_numpy().copy()
    age[index] = -1
    i = table.schema.get_field_index("sensor_age_steps")
    table = table.set_column(i, table.schema.field(i), pa.array(age))
    pq.write_table(table, path)
    resign(release)
    with pytest.raises(ValueError, match="age zero"):
        validate(release)


def test_tiny_negative_cost_cannot_hide_in_equation_tolerance(release):
    for item in read_json(release / "manifest.json")["files"]:
        if item["kind"] != "logged":
            continue
        path = release / item["path"]
        x = pq.read_table(path)
        outages = np.flatnonzero(~x["grid_available"].to_numpy())
        if not len(outages):
            continue
        index = outages[0]
        oracle_path = release / item["path"].replace("data/logged/", "data/oracle/")
        o = pq.read_table(oracle_path)
        for table, names, target in (
            (x, ["y_cost"], path),
            (o, [f"cf_cost_{a}" for a in range(3)], oracle_path),
        ):
            for name in names:
                data = table[name].to_numpy().copy()
                data[index] = -1e-8
                i = table.schema.get_field_index(name)
                table = table.set_column(i, table.schema.field(i), pa.array(data))
            pq.write_table(table, target)
        resign(release)
        with pytest.raises(ValueError, match="negative counterfactual cost"):
            validate(release)
        return
    pytest.fail("fixture must contain an outage")
