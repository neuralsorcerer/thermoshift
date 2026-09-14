# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Physical equations, data integrity and boundary conditions."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scipy.integrate import solve_ivp

from thermoshift import Config
from thermoshift.evaluation import evaluate_policy
from thermoshift.filesystem import atomic_json, read_json, sha256
from thermoshift.lifecycle import dataset_lock
from thermoshift.provenance import metadata_hashes, source_digest
from thermoshift.reading import iter_pairs, validated_manifest
from thermoshift.schema import SCHEMAS
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


def test_rc_equation_holds_across_decades_of_thermal_time_constant():
    # The test above samples the range a building actually occupies. thermal_step
    # is public and documented for any finite conductance and positive capacity,
    # so sweep eight decades of rate with a solver that tolerates the stiff end.
    rng = np.random.default_rng(936)
    for _ in range(40):
        temp, outdoor = rng.uniform(-40, 120, 2)
        g, c = 10 ** rng.uniform(-2, 2), 10 ** rng.uniform(-2, 2)
        heat, cooling = rng.uniform(-50, 500), rng.uniform(0, 500)
        solution = solve_ivp(
            lambda _, y, g=g, outdoor=outdoor, heat=heat, cooling=cooling, c=c: (
                (g * (outdoor - y) + heat - cooling) / c
            ),
            (0, 1),
            [temp],
            rtol=1e-12,
            atol=1e-14,
            method="LSODA",
        )
        assert solution.success
        # Relative: an equilibrium of outdoor + (heat - cooling) / g reaches far
        # from the ODE solver's absolute accuracy when conductance is small.
        assert thermal_step(temp, outdoor, g, c, heat, cooling) == pytest.approx(
            solution.y[0, -1], rel=1e-8, abs=1e-9
        )


def test_small_rate_series_joins_the_exponential_branch_without_a_step():
    # Below rate 1e-8 the heat response switches to a series expansion whose
    # truncation error is about rate^2/12. Holding indoor and outdoor temperature
    # equal cancels the ambient term, so an error in the response shows up one for
    # one rather than diluted. LSODA tracks the implementation to 2.6e-14 here.
    capacity, temp, outdoor, heat, cooling = 4.0, 0.0, 0.0, 6.0, 4.0
    rates = (1e-12, 1e-10, 1e-9, 5e-9, 9.9e-9, 1e-8, 1.1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 9e-5, 1e-3)
    for rate in rates:
        conductance = rate * capacity
        solution = solve_ivp(
            lambda _, y, conductance=conductance: (
                (conductance * (outdoor - y) + heat - cooling) / capacity
            ),
            (0, 1),
            [temp],
            rtol=1e-13,
            atol=1e-15,
            method="LSODA",
        )
        assert solution.success
        assert thermal_step(temp, outdoor, conductance, capacity, heat, cooling) == pytest.approx(
            solution.y[0, -1], rel=1e-11
        )
    series = thermal_step(temp, outdoor, np.nextafter(1e-8, 0) * capacity, capacity, heat, cooling)
    exponential = thermal_step(temp, outdoor, 1e-8 * capacity, capacity, heat, cooling)
    assert series == pytest.approx(exponential, rel=1e-12)


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


BUSY_PROBE = """
import sys
from thermoshift.lifecycle import dataset_lock
try:
    with dataset_lock(sys.argv[1]):
        print("acquired")
except ValueError as error:
    print(error)
"""


def test_a_busy_dataset_fails_immediately_instead_of_waiting(release):
    # README promises operations on a busy dataset "fail immediately with a busy
    # error", which comes from taking the lease non-blocking: measured, refusing
    # in 0.00s against never returning. The contention must cross processes -- in
    # one process the second acquire reports busy either way -- and the timeout
    # keeps a blocking lease a failure rather than a hung suite.
    import subprocess
    import sys

    with dataset_lock(release, generating=True):
        completed = subprocess.run(
            [sys.executable, "-c", BUSY_PROBE, str(release)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    assert "busy" in completed.stdout, (completed.stdout, completed.stderr)


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


def test_stress_splits_exhibit_their_documented_shifts(tmp_path):
    # Each bound below was chosen by removing the corresponding term from the
    # simulator and measuring what the statistic becomes; the mutated value is
    # quoted beside the bound. Sized so the stress splits carry enough rows for a
    # 0.4% outage rate to separate from the common one; a quarter of this does not.
    initialize(tmp_path, Config(rows=200_000, episode_steps=168, shard_buildings=400))
    generate(tmp_path, batch_buildings=200, progress=None)
    profiles = validate(tmp_path, write_report=False)["split_profiles"]
    common = ["train", "validation", "test"]

    def pooled(split_names, logged_column, oracle_column):
        values = []
        for split in split_names:
            with closing(iter_pairs(tmp_path, split=split, batch_size=65_536)) as batches:
                for _, x, o in batches:
                    table, name = (x, logged_column) if logged_column else (o, oracle_column)
                    values.append(table[name].to_numpy().astype(float))
        return np.concatenate(values)

    def average(field, split_names):
        return sum(profiles[split][field] for split in split_names) / len(split_names)

    # Splits under common conditions stay together, so a shift cannot be a
    # global change mistaken for a per-split one.
    outdoor = {split: profiles[split]["mean_outdoor_c"] for split in profiles}
    assert max(outdoor[s] for s in common) - min(outdoor[s] for s in common) < 1.0
    assert outdoor["test_sensor"] == pytest.approx(average("mean_outdoor_c", common), abs=1.0)

    # Heatwave: +8 degC outdoors (+0.4 without the offset).
    assert outdoor["test_heatwave"] - average("mean_outdoor_c", common) == pytest.approx(
        8.0, abs=1.0
    )

    # Heatwave: COP multiplied by 0.85. Comparing within a matched outdoor band
    # separates the multiplier from the temperature term the offset also drives.
    band_cop = {}
    for label, splits in (("common", common), ("heatwave", ["test_heatwave"])):
        outdoor_values = pooled(splits, "outdoor_temp_c", None)
        cop_values = pooled(splits, None, "cop")
        inside = (outdoor_values >= 30) & (outdoor_values <= 34)
        assert inside.sum() > 200, f"{label} sample is too small to compare"
        band_cop[label] = cop_values[inside].mean()
    # 0.98 once the multiplier is removed.
    assert band_cop["heatwave"] / band_cop["common"] == pytest.approx(0.85, abs=0.05)

    # Heatwave: a higher outage-entry probability (1.1x without it).
    assert profiles["test_heatwave"]["outage_rate"] > 2.5 * average("outage_rate", common)
    assert profiles["test_sensor"]["outage_rate"] == pytest.approx(
        average("outage_rate", common), abs=0.01
    )

    # Sensor: heavier and more persistent dropout (9.4x and 3.5x when either
    # the persistence or the entry probability is reduced to the common one).
    assert profiles["test_sensor"]["null_sensor_rate"] > 12 * average("null_sensor_rate", common)
    assert profiles["test_heatwave"]["null_sensor_rate"] == pytest.approx(
        average("null_sensor_rate", common), abs=0.02
    )

    # Sensor: drift scaled by 3 rather than 0.20 (ratio 0.9 without it).
    assert (
        pooled(["test_sensor"], None, "sensor_bias_c").std()
        > 5 * pooled(common, None, "sensor_bias_c").std()
    )


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


@pytest.mark.parametrize(
    "kind,column,hour,message",
    [
        ("logged", "solar_w_m2", 0, "outside daylight hours"),
        ("logged", "solar_w_m2", 12, "solar aperture changed"),
        ("oracle", "sensor_bias_c", 0, "sensor bias is not linear"),
        ("oracle", "sensor_bias_c", 12, "sensor bias is not linear"),
    ],
)
def test_columns_without_a_direct_equation_are_still_verified(release, kind, column, hour, message):
    # Irradiance and sensor bias feed no stored equation of their own. They are
    # pinned by the daylight schedule and by per-building invariants instead.
    manifest = read_json(release / "manifest.json")
    logged = next(f for f in manifest["files"] if f["kind"] == "logged")
    hours = pq.read_table(release / logged["path"])["hour"].to_numpy()
    index = int(np.flatnonzero(hours == hour)[0])
    relative = logged["path"]
    if kind == "oracle":
        relative = relative.replace("data/logged/", "data/oracle/")
    path = release / relative
    table = pq.read_table(path)
    values = table[column].to_numpy().copy()
    values[index] += 0.5
    i = table.schema.get_field_index(column)
    table = table.set_column(i, table.schema.field(i), pa.array(values))
    pq.write_table(table, path)
    resign(release)
    with pytest.raises(ValueError, match=message):
        validate(release)


@pytest.mark.parametrize(
    "kind,column,factor,offset,message",
    [
        ("logged", "floor_area_m2", 2.0, 0.0, "floor area outside its documented range"),
        ("oracle", "conductance_kw_per_c", 3.0, 0.0, "conductance_kw_per_c outside its documented"),
        ("oracle", "capacity_kwh_per_c", 3.0, 0.0, "capacity_kwh_per_c outside its documented"),
        ("oracle", "base_load_kw", 3.0, 0.0, "base_load_kw outside its documented"),
        ("oracle", "cop", 0.5, 0.0, "coefficient of performance below its nominal range"),
        ("logged", "occupancy_count", 1.0, 100.0, "occupancy above the building's capacity"),
    ],
)
def test_parameters_outside_their_documented_distribution_are_rejected(
    release, kind, column, factor, offset, message
):
    # Scaling a whole column keeps it constant within each trajectory and keeps
    # every stored equation self-consistent, so only the distribution checks can
    # notice. A release generated from a different building population looks
    # exactly like this.
    manifest = read_json(release / "manifest.json")
    relative = next(f["path"] for f in manifest["files"] if f["kind"] == "logged")
    if kind == "oracle":
        relative = relative.replace("data/logged/", "data/oracle/")
    path = release / relative
    table = pq.read_table(path)
    index = table.schema.get_field_index(column)
    values = table[column].to_numpy()
    changed = (values * factor + offset).astype(values.dtype)
    table = table.set_column(index, table.schema.field(index), pa.array(changed))
    pq.write_table(table, path)
    resign(release)
    with pytest.raises(ValueError, match=message):
        validate(release)


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


@pytest.mark.parametrize("hidden_confounding", [False, True])
def test_off_policy_estimates_recover_the_oracle_policy_value(tmp_path, hidden_confounding):
    # Both estimators target the oracle policy value on the logging state
    # distribution, so with the supplied true propensities they must land on it up
    # to sampling error. Measured over eight seeds the worst case is 0.68 standard
    # errors for IPS and 0.46 for SNIPS, against the three asserted below. Hidden
    # confounding is included because the supplied propensities stay valid under
    # it, unlike ones re-estimated from observed features.
    initialize(
        tmp_path,
        Config(rows=60_000, episode_steps=24, seed=42, hidden_confounding=hidden_confounding),
    )
    generate(tmp_path, progress=None)
    finalize(tmp_path)

    result = evaluate_policy(tmp_path, split="train")
    truth = result["oracle_policy_value"]["value"]
    assert result["overlap_status"] == "observed_matches"
    for name in ("ips", "snips"):
        estimate = result[name]
        assert abs(estimate["value"] - truth) < 3 * estimate["se_building_cluster"]
        low, high = estimate["ci95_normal_approx"]
        assert low < estimate["value"] < high

    # The logged policy is a different policy, so its value is a different number;
    # otherwise the comparison above would hold for any estimator that simply
    # averaged the factual rewards.
    assert result["logged_policy_value"]["value"] != pytest.approx(truth, abs=1e-6)
    # Regret is measured against the best action available, so it cannot be negative.
    assert result["oracle_one_step_regret"]["value"] >= 0


def test_every_stored_column_is_pinned_by_some_check(tmp_path):
    """Changing any one stored value anywhere must fail validation.

    The release holds a whole number of trajectories deliberately. A final
    trajectory truncated to a single daylight hour leaves its irradiance pinned
    only by the documented 550-900 W/m2 band, because the aperture comparison
    needs two lit hours of the same building and against a dark neighbour it
    multiplies out to nothing. A sub-percent change then stays inside the band.
    The aperture is not stored and the clear-sky factor is redrawn every hour, so
    one lit observation identifies neither: a limit of the stored columns rather
    than a missing check.
    """
    initialize(tmp_path, Config(rows=216, episode_steps=24))
    generate(tmp_path, progress=None)
    finalize(tmp_path)

    undetected = []
    for kind, schema in SCHEMAS.items():
        item = next(f for f in read_json(tmp_path / "manifest.json")["files"] if f["kind"] == kind)
        path = tmp_path / item["path"]
        pristine = path.read_bytes()
        table = pq.read_table(path)
        rows = {"first": 0, "interior": table.num_rows // 2, "last": table.num_rows - 1}
        for column in schema.names:
            index = schema.get_field_index(column)
            field = schema.field(index)
            original = table[column].to_numpy(zero_copy_only=False)
            for where, row in rows.items():
                values = original.copy()
                values[row] = (
                    (not values[row]) if pa.types.is_boolean(field.type) else values[row] + 1
                )
                pq.write_table(
                    table.set_column(index, field, pa.array(values, type=field.type)), path
                )
                resign(tmp_path)
                try:
                    validate(tmp_path)
                    undetected.append(f"{kind}.{column} at the {where} row")
                except ValueError:
                    pass
                path.write_bytes(pristine)
        resign(tmp_path)
    assert not undetected, "corruptions that validation did not detect: " + ", ".join(undetected)
