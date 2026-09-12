# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pyarrow as pa
import pytest

from thermoshift import Config, evaluation
from thermoshift.simulator import simulate


def test_cluster_variance_equals_independent_direct_calculation():
    ids = np.array([1, 1, 2, 2, 2, 3, 3])
    rewards = np.array([1.0, 3.0, 5.0, -2.0, 1.0, 4.0, 5.0])
    weights = np.array([1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0])
    values = np.column_stack(
        [np.ones(len(ids)), rewards * weights, weights, rewards, rewards, rewards]
    )
    moments = evaluation.ClusterMoments()
    # Deliberately split a building across chunks.
    moments.add(ids[:3], values[:3])
    moments.add(ids[3:6], values[3:6])
    moments.add(ids[6:], values[6:])
    moments.flush()
    groups = np.array([values[ids == g].sum(axis=0) for g in [1, 2, 3]])
    estimate = groups[:, 1].sum() / groups[:, 2].sum()
    se = np.sqrt(3 / 2 * np.sum((groups[:, 1] - estimate * groups[:, 2]) ** 2)) / groups[:, 2].sum()
    result = moments.ratio(1, 2)
    assert moments.count == 3
    assert result["value"] == pytest.approx(estimate)
    assert result["se_building_cluster"] == pytest.approx(se)


def test_no_policy_overlap_is_reported_without_dividing_by_zero():
    moments = evaluation.ClusterMoments()
    moments.add(
        np.array([1, 2]), np.array([[1.0, 0.0, 0.0, 2.0, 2.0, 0.0], [1.0, 0.0, 0.0, 4.0, 4.0, 0.0]])
    )
    moments.flush()
    assert moments.ratio(1, 2)["value"] is None
    assert moments.ratio(1, 0)["value"] == 0.0


def test_large_offset_does_not_cancel_cluster_variance():
    m = evaluation.ClusterMoments()
    values = np.column_stack(
        [np.ones(100), 1e12 + np.arange(100), np.ones(100), np.zeros((100, 3))]
    )
    m.add(np.arange(100), values)
    m.flush()
    assert m.ratio(1, 0)["se_building_cluster"] == pytest.approx(
        np.std(np.arange(100), ddof=1) / 10, rel=1e-12
    )


def test_cluster_order_and_empty_input():
    m = evaluation.ClusterMoments()
    m.add(np.array([], dtype=int), np.empty((0, 6)))
    with pytest.raises(ValueError, match="ordered"):
        m.add(np.array([2, 1]), np.ones((2, 6)))
    m.add(np.array([1]), np.ones((1, 6)))
    m.flush()
    with pytest.raises(ValueError, match="reopen"):
        m.add(np.array([1]), np.ones((1, 6)))


def test_no_overlap_has_no_misleading_zero_width_ips_interval():
    m = evaluation.ClusterMoments()
    m.add(
        np.array([1, 2]), np.array([[1.0, 0.0, 0.0, 2.0, 2.0, 0.0], [1.0, 0.0, 0.0, 4.0, 4.0, 0.0]])
    )
    m.flush()
    assert m.ratio(1, 0)["ci95_normal_approx"] is None


@pytest.fixture
def policy_tables():
    logged, oracle, _ = simulate(Config(rows=96, episode_steps=24), 0, 4)
    return logged, oracle


def test_policy_estimates_match_direct_building_calculation(monkeypatch, policy_tables):
    logged, oracle = policy_tables

    def batches(*args, **kwargs):
        for start in range(0, len(logged), 17):
            yield {}, logged.slice(start, 17), oracle.slice(start, 17)

    monkeypatch.setattr(evaluation, "iter_pairs", batches)
    result = evaluation.evaluate_policy("unused")
    gap = logged["obs_temp_last_c"].to_numpy().astype(float) - logged[
        "setpoint_c"
    ].to_numpy().astype(float)
    actions = np.where(gap > 1.5, 2, np.where(gap > -0.3, 1, 0))
    weights = (actions == logged["action"].to_numpy()) / logged["propensity"].to_numpy()
    factual = logged["y_reward"].to_numpy().astype(float)
    rewards = np.column_stack([oracle[f"cf_reward_{a}"].to_numpy() for a in range(3)])
    truth = rewards[np.arange(len(actions)), actions]
    numerators = [weights * factual, weights * factual, truth, factual, rewards.max(1) - truth]
    denominators = [np.ones(len(logged)), weights] + [np.ones(len(logged))] * 3
    names = ["ips", "snips", "oracle_policy_value", "logged_policy_value", "oracle_one_step_regret"]
    ids = logged["building_id"].to_numpy()
    for name, numerator, denominator in zip(names, numerators, denominators):
        value = numerator.sum() / denominator.sum()
        residuals = np.array(
            [
                (numerator[ids == bid] - value * denominator[ids == bid]).sum()
                for bid in np.unique(ids)
            ]
        )
        se = np.sqrt(4 / 3 * (residuals**2).sum()) / denominator.sum()
        assert result[name]["value"] == pytest.approx(value)
        assert result[name]["se_building_cluster"] == pytest.approx(se)
    assert result["rows"] == 96 and result["independent_buildings"] == 4
    assert result["row_weight_ess_diagnostic"] == pytest.approx(
        weights.sum() ** 2 / (weights**2).sum()
    )


@pytest.mark.parametrize(
    "kind,column,value,message",
    [
        ("logged", "obs_temp_last_c", np.nan, "nonfinite policy input"),
        ("logged", "setpoint_c", np.inf, "nonfinite policy input"),
        ("logged", "action", 3, "invalid logged action"),
        ("logged", "propensity", 1.5, "invalid logging propensity"),
        ("logged", "propensity", 0.0, "invalid logging propensity"),
        ("logged", "propensity", 0.12345, "selected propensity mismatch"),
        ("logged", "p_action_0", -0.1, "invalid logging probabilities"),
        ("logged", "y_reward", 123.0, "factual reward mismatch"),
        ("oracle", "cf_reward_0", -np.inf, "nonfinite oracle reward"),
        ("oracle", "building_id", 99, "oracle alignment fails"),
        ("oracle", "step", 99, "oracle alignment fails"),
    ],
)
def test_invalid_policy_data_fails_and_closes_reader(
    monkeypatch, policy_tables, kind, column, value, message
):
    logged, oracle = policy_tables
    table = logged if kind == "logged" else oracle
    field = table.schema.field(column)
    changed = table[column].to_pylist()
    changed[0] = value
    table = table.set_column(
        table.schema.get_field_index(column), field, pa.array(changed, type=field.type)
    )
    if kind == "logged":
        logged = table
    else:
        oracle = table
    closed = []

    def batches(*args, **kwargs):
        try:
            yield {}, logged, oracle
        finally:
            closed.append(True)

    monkeypatch.setattr(evaluation, "iter_pairs", batches)
    with pytest.raises(ValueError, match=message):
        evaluation.evaluate_policy("unused")
    assert closed == [True]


@pytest.mark.parametrize("split", [None, "unknown", 1, []])
def test_policy_rejects_invalid_split_before_reading(monkeypatch, split):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("invalid split reached the reader")

    monkeypatch.setattr(evaluation, "iter_pairs", unexpected_read)
    with pytest.raises(ValueError, match="unknown split"):
        evaluation.evaluate_policy("unused", split=split)
