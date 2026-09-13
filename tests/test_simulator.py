# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pyarrow as pa
import pytest

from thermoshift import Config, simulate
from thermoshift.random import split_codes
from thermoshift.simulator import thermal_step


def test_thermal_solution_matches_hand_calculation():
    # C/G=2 h, no forcing: the 10 degC gap decays by exp(-1/2).
    result = thermal_step(20.0, 30.0, 2.0, 4.0, 0.0, 0.0)
    assert result == pytest.approx(30 - 10 * np.exp(-0.5))
    assert thermal_step(25.0, 25.0, 2.0, 4.0, 3.0, 3.0) == pytest.approx(25.0)


@pytest.mark.parametrize("conductance", [0.0, 1e-310, np.nextafter(0.0, 1.0)])
def test_thermal_solution_at_insulated_limit(conductance):
    # With no ambient transfer, 3 kW heats 4 kWh/degC by 0.75 degC in one hour.
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        result = thermal_step(20.0, 30.0, conductance, 4.0, 3.0, 0.0)
    assert result == pytest.approx(20.75, abs=1e-12)


def test_thermal_solution_promotes_and_broadcasts_array_inputs():
    conductance = np.array([[0], [2]], dtype=np.uint64)
    result = thermal_step([20, 25], 30, conductance, 4, [3, 0], 0)
    expected = np.array([[20.75, 25.0], [31.5 + (20 - 31.5) * np.exp(-0.5), 30 - 5 * np.exp(-0.5)]])
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "conductance,capacity",
    [(-1, 4), (1, 0), (1, -4), (np.nan, 4), (np.inf, 4), (1, np.nan), (1, np.inf)],
)
def test_thermal_solution_rejects_invalid_physical_parameters(conductance, capacity):
    with pytest.raises(ValueError, match="conductance.*capacity"):
        thermal_step(20, 30, conductance, capacity, 3, 0)


@pytest.mark.parametrize(
    "arguments",
    [
        (np.nan, 30, 2, 4, 3, 0),
        (20, np.inf, 2, 4, 3, 0),
        (20, 30, 2, 4, -np.inf, 0),
        (20, 30, 2, 4, 3, np.nan),
    ],
)
def test_thermal_solution_rejects_nonfinite_forcing(arguments):
    with pytest.raises(ValueError, match="must be finite"):
        thermal_step(*arguments)


def test_thermal_solution_reports_incompatible_shapes():
    with pytest.raises(ValueError, match="broadcast-compatible"):
        thermal_step(np.zeros(2), np.zeros(3), 2, 4, 3, 0)


def test_thermal_solution_handles_overflowing_rate():
    # With C/G far below one hour, temperature reaches the forced equilibrium.
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        result = thermal_step(20, 30, 1e200, 1e-200, 3e200, 0)
    assert result == pytest.approx(33.0)


def test_thermal_solution_recovers_representable_result_after_intermediate_overflow():
    result = thermal_step(1e308, -1e308, 1, 1, -1e308, 1e308)
    alpha = -np.expm1(np.longdouble(-1))
    expected = (
        np.longdouble(1e308)
        + alpha * (np.longdouble(-1e308) - np.longdouble(1e308))
        + (np.longdouble(-1e308) - np.longdouble(1e308)) * alpha
    )
    assert result == pytest.approx(float(expected))


def test_thermal_solution_rejects_truly_nonfinite_result_from_finite_inputs():
    with pytest.raises(ValueError, match="produce a nonfinite temperature"):
        thermal_step(1e308, 1e308, 0, 1, 1e308, -1e308)


def test_batch_size_does_not_change_numeric_records():
    config = Config(rows=168 * 41 - 7)
    large_x, large_o, large_s = simulate(config, 0, config.buildings)
    pieces = [
        simulate(config, lo, min(lo + 7, config.buildings)) for lo in range(0, config.buildings, 7)
    ]
    assert large_x.equals(pa.concat_tables([x for x, _, _ in pieces]))
    assert large_o.equals(pa.concat_tables([o for _, o, _ in pieces]))
    np.testing.assert_array_equal(large_s, np.concatenate([s for _, _, s in pieces]))


def test_simulate_requires_config_instance_before_accessing_it():
    with pytest.raises(TypeError, match="Config instance"):
        simulate({}, 0, 1)


def test_ids_remain_int64_beyond_two_billion_rows():
    config = Config(rows=3_000_000_017)
    x, _, _ = simulate(config, config.buildings - 2, config.buildings)
    ids = x["row_id"].to_numpy()
    assert ids.dtype == np.int64
    assert ids[-1] == config.rows - 1
    assert np.all(np.diff(ids) == 1)
    assert x["episode_end"][-1].as_py() is True


def test_prefix_stability_across_dataset_sizes_except_end_marker():
    a, b = Config(rows=1503), Config(rows=2500)
    xa, oa, sa = simulate(a, 0, a.buildings)
    xb, ob, sb = simulate(b, 0, a.buildings)
    assert xa.drop(["episode_end"]).equals(xb.slice(0, 1503).drop(["episode_end"]))
    assert oa.equals(ob.slice(0, 1503))
    np.testing.assert_array_equal(sa, sb[:1503])


def test_split_assignment_is_disjoint_and_stable():
    ids = np.arange(10000)
    a = split_codes(ids, 42)
    np.testing.assert_array_equal(a[:2000], split_codes(ids[:2000], 42))
    assert set(a.tolist()) == set(range(5))
    assert np.mean(a == 0) == pytest.approx(0.7, abs=0.025)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 0},
        {"rows": -1},
        {"rows": 1.5},
        {"seed": -1},
        {"episode_steps": 1},
        {"exploration": 0},
        {"exploration": float("nan")},
        {"comfort_weight": float("nan")},
        {"shard_buildings": 0},
        {"compression": "none"},
        {"hidden_confounding": 1},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


@pytest.mark.parametrize("field,value", [("compression", "zstd"), ("schema_version", "2.0")])
def test_configuration_rejects_array_string_values(field, value):
    # Single-element arrays compare equal to strings but cannot be fingerprinted.
    with pytest.raises(ValueError):
        Config(**{field: np.array([value])})
