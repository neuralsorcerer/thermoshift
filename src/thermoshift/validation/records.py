# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Streaming checks for row identity, trajectories, logging and physical outcomes."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa

from thermoshift._checks import array, close, require
from thermoshift.config import SPLITS, Config
from thermoshift.random import split_codes
from thermoshift.reading import _iter_pairs
from thermoshift.schema import SCHEMAS


def _required_values(x: pa.Table, o: pa.Table) -> None:
    """Check nullability and finite numeric values."""
    for kind, table in (("logged", x), ("oracle", o)):
        for f in SCHEMAS[kind]:
            col = table[f.name]
            if not f.nullable:
                require(col.null_count == 0, f"unexpected null: {f.name}")
            if pa.types.is_floating(f.type):
                require(np.isfinite(col.drop_null().to_numpy()), f"nonfinite values: {f.name}")


def _identifiers(config: Config, item: dict, x: pa.Table, o: pa.Table) -> None:
    """Check identity, split membership and time labels."""
    rid, bid, step = (array(x, k) for k in ("row_id", "building_id", "step"))
    for name in ("row_id", "building_id", "step"):
        require(array(x, name) == array(o, name), "logged/oracle join mismatch")
    low, high = config.bounds(item["shard_id"])
    require((bid >= low) & (bid < high), "building outside shard interval")
    require((rid >= 0) & (rid < config.rows), "row ID out of bounds")
    require((step >= 0) & (step < config.episode_steps), "invalid step")
    require(rid == bid * config.episode_steps + step, "row ID equation fails")
    require(np.diff(rid) > 0, "duplicate or unsorted row ID")
    require(
        split_codes(bid, config.seed) == SPLITS.index(item["split"]),
        "building leakage across splits",
    )
    require(array(x, "hour") == step % 24, "hour/step mismatch")
    require(
        (array(x, "day_of_week") >= 0) & (array(x, "day_of_week") <= 6), "weekday out of bounds"
    )
    require(np.isin(array(x, "building_type"), [0, 1]), "building type out of bounds")
    require(
        array(x, "episode_end") == ((step == config.episode_steps - 1) | (rid == config.rows - 1)),
        "incorrect episode boundary",
    )


def _trajectory(
    x: pa.Table, o: pa.Table, previous_x: pa.Table | None, previous_o: pa.Table | None
) -> None:
    """Check state and sensor continuity across reading batches."""
    step = array(x, "step")
    joined_x = pa.concat_tables([previous_x, x]) if previous_x is not None else x
    joined_o = pa.concat_tables([previous_o, o]) if previous_o is not None else o
    joined_bid = array(joined_x, "building_id")
    same = np.diff(joined_bid) == 0
    require(np.diff(array(joined_x, "row_id")) > 0, "duplicate row across batches")
    require(np.diff(array(joined_x, "step"))[same] == 1, "nonconsecutive episode steps")
    require(array(joined_x, "step")[1:][~same] == 0, "episode does not start at step zero")
    require(array(joined_x, "episode_end")[:-1] == ~same, "incorrect episode transition")
    if previous_x is None:
        require(step[0] == 0, "file starts inside an episode")
    require(
        array(joined_o, "true_temp_c")[1:][same] == array(joined_x, "y_next_temp_c")[:-1][same],
        "state continuity fails",
    )
    for name in ("building_type", "floor_area_m2", "hvac_capacity_kw", "setpoint_c"):
        v = array(joined_x, name)
        require(v[1:][same] == v[:-1][same], f"building parameter changed: {name}")
    for name in ("conductance_kw_per_c", "capacity_kwh_per_c"):
        v = array(joined_o, name)
        require(v[1:][same] == v[:-1][same], f"building parameter changed: {name}")
    # Solar gain is the irradiance scaled by a constant per-building aperture, so
    # the cross-multiplied ratio cannot move within a trajectory. This is the one
    # check that ties stored irradiance to stored solar heat.
    irradiance = array(joined_x, "solar_w_m2").astype(float)
    solar_heat = array(joined_o, "solar_heat_kw").astype(float)
    close(
        (solar_heat[1:] * irradiance[:-1])[same],
        (solar_heat[:-1] * irradiance[1:])[same],
        "solar aperture changed within a trajectory",
    )
    # Sensor bias is affine in the step index, so its second difference vanishes.
    drift = np.diff(array(joined_o, "sensor_bias_c").astype(float))
    consecutive = same[1:] & same[:-1]
    close(
        drift[1:][consecutive],
        drift[:-1][consecutive],
        "sensor bias is not linear in the step index",
    )
    days = array(joined_x, "day_of_week").astype(int)
    new_day = array(joined_x, "hour")[1:] == 0
    require(days[1:][same] == ((days[:-1] + new_day) % 7)[same], "weekday progression fails")
    sensor = array(joined_x, "obs_temp_c")
    last = array(joined_x, "obs_temp_last_c")
    ages = array(joined_x, "sensor_age_steps")
    absent = np.isnan(sensor)
    continuing_dropout = absent[1:] & same
    require(
        last[1:][continuing_dropout] == last[:-1][continuing_dropout],
        "sensor history changed during dropout",
    )
    require(
        ages[1:][continuing_dropout] == ages[:-1][continuing_dropout] + 1,
        "sensor age did not increment",
    )
    initial_dropout = (array(joined_x, "step") == 0) & absent
    require(
        last[initial_dropout] == array(joined_x, "setpoint_c")[initial_dropout],
        "initial sensor fallback mismatch",
    )
    require(ages[initial_dropout] == 1, "initial sensor age mismatch")


def _logging(config: Config, x: pa.Table, o: pa.Table) -> None:
    """Check action probabilities and decision-time sensing."""
    temp = array(o, "true_temp_c")
    action = array(x, "action").astype(int)
    require((action >= 0) & (action < 3), "invalid action")
    probs = np.column_stack([array(x, f"p_action_{a}") for a in range(3)])
    require((probs >= config.exploration / 3) & (probs <= 1), "positivity violation")
    close(probs.sum(axis=1), 1, "probabilities do not sum to 1", atol=2e-7)
    require(
        array(x, "propensity") == probs[np.arange(len(action)), action],
        "selected propensity mismatch",
    )
    policy_temp = array(x, "obs_temp_last_c").astype(float)
    if config.hidden_confounding:
        policy_temp += 0.8 * (temp - policy_temp)
    demand = np.clip(
        (
            policy_temp
            - array(x, "setpoint_c").astype(float)
            + 0.15
            * (array(x, "outdoor_temp_c").astype(float) - array(x, "setpoint_c").astype(float))
        )
        / 3,
        0,
        1,
    )
    levels = np.array([0, 0.5, 1])
    logits = (
        -5 * (levels - demand[:, None]) ** 2
        - 0.7
        * array(x, "price_per_kwh").astype(float)[:, None]
        * array(x, "hvac_capacity_kw").astype(float)[:, None]
        * levels
    )
    logits -= logits.max(axis=1, keepdims=True)
    policy_probs = np.exp(logits)
    policy_probs /= policy_probs.sum(axis=1, keepdims=True)
    policy_probs = (1 - config.exploration) * policy_probs + config.exploration / 3
    if config.hidden_confounding:
        close(probs, policy_probs, "logging policy equation fails", atol=6e-7)
    else:
        require(probs == policy_probs, "observable logging policy is not exactly reproducible")
    nulls = array(x, "obs_temp_c")
    missing = np.isnan(nulls)
    require(missing == (array(x, "sensor_age_steps") > 0), "sensor null/age mismatch")
    require(
        array(x, "sensor_age_steps")[~missing] == 0, "available sensor reading must have age zero"
    )
    require(
        array(x, "obs_temp_last_c")[~missing] == nulls[~missing],
        "latest available sensor value differs",
    )
    require(
        array(o, "sensor_fault") == (missing | (np.abs(array(o, "sensor_bias_c")) > 0.75)),
        "sensor fault label mismatch",
    )


def _domains(x: pa.Table, o: pa.Table) -> None:
    """Check physical parameter domains."""
    for name in ("floor_area_m2", "hvac_capacity_kw", "price_per_kwh", "carbon_kg_per_kwh"):
        require(array(x, name) > 0, f"nonpositive {name}")
    for name in ("conductance_kw_per_c", "capacity_kwh_per_c", "cop", "base_load_kw"):
        require(array(o, name) > 0, f"nonpositive {name}")
    for name in ("internal_heat_kw", "solar_heat_kw"):
        require(array(o, name) >= 0, f"negative {name}")
    require(array(x, "occupancy_count") >= 0, "negative occupancy")
    require(array(x, "solar_w_m2") >= 0, "negative irradiance")
    # Irradiance follows the documented daylight schedule: exactly zero whenever
    # the daylight sine vanishes, and between 550 and 900 W/m2 per unit sine
    # otherwise. Bounds are relative so they hold at the tiny dusk sine too.
    hour = array(x, "hour").astype(float)
    shape = np.where(
        (hour >= 6) & (hour <= 18), np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12)), 0.0
    )
    irradiance = array(x, "solar_w_m2").astype(float)
    dark = shape == 0
    require(irradiance[dark] == 0, "nonzero irradiance outside daylight hours")
    lit = ~dark
    require(
        (irradiance[lit] >= 550 * shape[lit] * (1 - 1e-6))
        & (irradiance[lit] <= 900 * shape[lit] * (1 + 1e-6)),
        "irradiance outside its daylight envelope",
    )
    _distributions(
        x, o, area=array(x, "floor_area_m2").astype(float), irradiance=irradiance, lit=lit
    )


def _within(values, low, high, message):
    """Bound a documented open interval, with room for float32 serialization."""
    require((values > low * (1 - 1e-6)) & (values < high * (1 + 1e-6)), message)


def _distributions(x: pa.Table, o: pa.Table, area, irradiance, lit) -> None:
    """Check the per-building draws the model reference documents.

    These are the only checks that look at where a parameter came from rather
    than how the stored columns relate. Every equation elsewhere is satisfied by
    a release generated from a different set of distributions, so without these
    a changed building population validates cleanly.
    """
    _within(area, 50, 300, "floor area outside its documented range")
    _within(array(x, "setpoint_c").astype(float), 23, 25, "setpoint outside its documented range")
    for name, table, low, high in (
        ("conductance_kw_per_c", o, 0.0025 * 0.65, 0.0025 * 1.35),
        ("capacity_kwh_per_c", o, 0.040 * 0.70, 0.040 * 1.30),
        ("hvac_capacity_kw", x, 0.025 * 0.80, 0.025 * 1.20),
        ("base_load_kw", o, 0.003 * 0.35, 0.003 * 1.00),
    ):
        _within(
            array(table, name).astype(float) / area,
            low,
            high,
            f"{name} outside its documented share of floor area",
        )
    if lit.any():
        # Solar heat is irradiance/1000 times the aperture, so the aperture is
        # recoverable wherever the sun is up.
        aperture = array(o, "solar_heat_kw").astype(float)[lit] * 1000 / irradiance[lit]
        _within(
            aperture / area[lit], 0.035, 0.100, "solar aperture outside its share of floor area"
        )
    # Occupancy is a draw from the building's capacity, or zero.
    require(
        array(x, "occupancy_count") <= 1 + np.floor(area / 35 + 1e-6),
        "occupancy above the building's capacity",
    )
    # Actual COP is the nominal draw reduced by 0.045 per degree above 25, then
    # clipped and, in the heatwave split, scaled by 0.85. Dividing by that scale
    # on one side and not the other brackets the nominal draw without needing to
    # know which split the row came from.
    excess = np.maximum(array(x, "outdoor_temp_c").astype(float) - 25, 0)
    cop = array(o, "cop").astype(float)
    require(
        cop / 0.85 + 0.045 * excess > 3.2 * (1 - 1e-6),
        "coefficient of performance below its nominal range",
    )
    require(
        cop + 0.045 * excess < 4.0 * (1 + 1e-6),
        "coefficient of performance above its nominal range",
    )


def _outcomes(config: Config, x: pa.Table, o: pa.Table) -> None:
    """Check potential outcomes, accounting and the thermal equation."""
    action = array(x, "action").astype(int)
    temp = array(o, "true_temp_c")
    cf = {}
    for name in ("next_temp_c", "energy_kwh", "cost", "carbon_kg", "discomfort_c", "reward"):
        cf[name] = np.column_stack([array(o, f"cf_{name}_{a}") for a in range(3)])
        require(
            array(x, f"y_{name}") == cf[name][np.arange(len(action)), action],
            f"factual/counterfactual mismatch: {name}",
        )
    for name in ("energy_kwh", "cost", "carbon_kg", "discomfort_c"):
        require(cf[name] >= 0, f"negative counterfactual {name}")
    require(cf["reward"] <= 0, "positive counterfactual reward")
    require(np.diff(cf["next_temp_c"], axis=1) <= 1e-5, "more cooling increased temperature")
    require(np.diff(cf["energy_kwh"], axis=1) >= -1e-6, "energy not monotone in cooling input")
    grid = array(x, "grid_available")
    for name, potential in cf.items():
        require(potential[~grid] == potential[~grid, :1], f"outage branches differ: {name}")
    require(cf["energy_kwh"][~grid] == 0, "outage has served energy")
    level = np.array([0, 0.5, 1])
    expected_energy = grid[:, None] * (
        array(o, "base_load_kw")[:, None] + array(x, "hvac_capacity_kw")[:, None] * level
    )
    close(
        array(o, "internal_heat_kw"),
        0.12 * array(x, "occupancy_count") + grid * array(o, "base_load_kw"),
        "background heat accounting fails",
    )
    close(cf["energy_kwh"], expected_energy, "electrical accounting fails")
    require(cf["energy_kwh"] >= 0, "negative served energy")
    close(
        cf["cost"], cf["energy_kwh"] * array(x, "price_per_kwh")[:, None], "cost accounting fails"
    )
    close(
        cf["carbon_kg"],
        cf["energy_kwh"] * array(x, "carbon_kg_per_kwh")[:, None],
        "carbon accounting fails",
    )
    # Independently expressed equilibrium/exponential form of the ODE.
    heat = sum(
        array(o, k).astype(float) for k in ("internal_heat_kw", "solar_heat_kw", "process_heat_kw")
    )
    conductance = array(o, "conductance_kw_per_c").astype(float)[:, None]
    capacity = array(o, "capacity_kwh_per_c").astype(float)[:, None]
    cooling = (
        array(x, "hvac_capacity_kw").astype(float)[:, None]
        * level
        * grid[:, None]
        * array(o, "cop").astype(float)[:, None]
    )
    equilibrium = (
        array(x, "outdoor_temp_c").astype(float)[:, None] + (heat[:, None] - cooling) / conductance
    )
    expected_temp = equilibrium + (temp.astype(float)[:, None] - equilibrium) * np.exp(
        -conductance / capacity
    )
    close(cf["next_temp_c"], expected_temp, "thermal balance fails", atol=3e-5)
    discomfort = np.maximum(np.abs(cf["next_temp_c"] - array(x, "setpoint_c")[:, None]) - 1, 0)
    close(cf["discomfort_c"], discomfort, "comfort proxy mismatch", atol=1e-5)
    expected_reward = -(
        cf["cost"]
        + config.carbon_weight * cf["carbon_kg"]
        + config.comfort_weight
        * np.where(array(x, "occupancy_count") > 0, 1, 0.1)[:, None]
        * cf["discomfort_c"] ** 2
    )
    close(cf["reward"], expected_reward, "reward equation fails")
    best = array(o, "oracle_action").astype(int)
    require((best >= 0) & (best < 3), "invalid oracle action")
    require(best == cf["reward"].argmax(axis=1), "oracle action or tie-breaking is incorrect")


def scan_records(root: Path, config: Config, manifest: dict, batch_size: int) -> dict:
    """Check every batch and aggregate per-split profiles."""
    report = {"split_profiles": {}}
    profiles = defaultdict(
        lambda: {
            "rows": 0,
            "sensor_nulls": 0,
            "outage_rows": 0,
            "outdoor_sum": 0.0,
            "energy_sum": 0.0,
            "reward_sum": 0.0,
            "regret_sum": 0.0,
            "temp_min": float("inf"),
            "temp_max": float("-inf"),
            "action_counts": np.zeros(3, dtype=np.int64),
        }
    )
    previous_path, previous_x, previous_o = None, None, None
    for item, x, o in _iter_pairs(root, manifest, batch_size=batch_size):
        if previous_path != item["path"]:
            if previous_x is not None:
                require(array(previous_x, "episode_end")[0], "file ends inside an episode")
            previous_x, previous_o, previous_path = None, None, item["path"]
        _required_values(x, o)
        _identifiers(config, item, x, o)
        _trajectory(x, o, previous_x, previous_o)
        _logging(config, x, o)
        _domains(x, o)
        _outcomes(config, x, o)
        previous_x, previous_o = x.slice(x.num_rows - 1), o.slice(o.num_rows - 1)
        action = array(x, "action").astype(int)
        missing = np.isnan(array(x, "obs_temp_c"))
        grid = array(x, "grid_available")
        temp = array(o, "true_temp_c")
        cf = {"reward": np.column_stack([array(o, f"cf_reward_{a}") for a in range(3)])}
        p = profiles[item["split"]]
        p["rows"] += len(action)
        p["sensor_nulls"] += int(missing.sum())
        p["outage_rows"] += int((~grid).sum())
        p["outdoor_sum"] += float(array(x, "outdoor_temp_c").sum(dtype=np.float64))
        p["energy_sum"] += float(array(x, "y_energy_kwh").sum(dtype=np.float64))
        p["reward_sum"] += float(array(x, "y_reward").sum(dtype=np.float64))
        p["regret_sum"] += float(
            (cf["reward"].max(axis=1) - array(x, "y_reward")).sum(dtype=np.float64)
        )
        p["temp_min"] = min(p["temp_min"], float(temp.min()))
        p["temp_max"] = max(p["temp_max"], float(temp.max()))
        p["action_counts"] += np.bincount(action, minlength=3)
    require(
        previous_x is not None and array(previous_x, "episode_end")[0],
        "final file ends inside an episode",
    )
    for split, p in profiles.items():
        n = p["rows"]
        require(n == manifest["split_rows"][split], "scanned row count mismatch")
        report["split_profiles"][split] = {
            "rows": n,
            "null_sensor_rate": p["sensor_nulls"] / n,
            "outage_rate": p["outage_rows"] / n,
            "mean_outdoor_c": p["outdoor_sum"] / n,
            "mean_energy_kwh": p["energy_sum"] / n,
            "mean_logged_reward": p["reward_sum"] / n,
            "mean_one_step_regret": p["regret_sum"] / n,
            "true_temp_min_c": p["temp_min"],
            "true_temp_max_c": p["temp_max"],
            "action_counts": p["action_counts"].tolist(),
        }
    report["checks"] = [
        "complete row coverage",
        "schema and nullability",
        "SHA-256 integrity",
        "unique IDs and disjoint building splits",
        "trajectory continuity",
        "action positivity",
        "factual-counterfactual consistency",
        "thermal/electrical balance",
        "cost/carbon/reward equations",
        "cooling monotonicity",
        "daylight irradiance envelope",
        "documented building parameter distributions",
        "constant solar aperture and linear sensor drift",
        "sensor missingness and fault labels",
        "oracle one-step optimality",
    ]
    return report
