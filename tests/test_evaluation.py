# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from fractions import Fraction

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


def test_anchoring_survives_a_large_offset_with_uneven_denominators():
    # Uneven weights leave a different large common-offset contribution in each building.
    rng = np.random.default_rng(7)
    clusters, per_cluster = 8, 5
    ids = np.repeat(np.arange(clusters), per_cluster)
    n = clusters * per_cluster
    rewards = 1e12 + rng.normal(0, 1e3, n)
    weights = rng.gamma(2, 2, n)
    values = np.column_stack(
        [np.ones(n), weights * rewards, weights, rewards, rewards, np.abs(rng.normal(0, 1, n))]
    )

    # Exact rational arithmetic decides what the answer is.
    totals = [
        (
            sum(Fraction(v) for v in values[ids == g][:, 1]),
            sum(Fraction(v) for v in values[ids == g][:, 2]),
        )
        for g in range(clusters)
    ]
    denominator = sum(d for _, d in totals)
    ratio = sum(z for z, _ in totals) / denominator
    residual_ss = sum((z - ratio * d) ** 2 for z, d in totals)
    exact = math.sqrt(float(residual_ss * Fraction(clusters, clusters - 1) / denominator**2))

    def run(anchored):
        moments = evaluation.ClusterMoments()
        if not anchored:
            # Disable anchor updates to leave every numerator unshifted.
            moments.anchor_denominator[:] = np.inf
        for start, stop in ((0, 13), (13, 29), (29, n)):  # clusters split across chunks
            moments.add(ids[start:stop], values[start:stop])
        moments.flush()
        return moments

    # The tolerance is the floor of the arithmetic. Each cluster total is ~2e13
    # against a residual of ~9e3, so the subtraction keeps only ULP(2e13) / 9e3 ~
    # 2e-7 however it is arranged. Measured across 200 seeds the worst case is
    # 5.7e-7 and the constant moves about 2x between platforms.
    anchored = run(anchored=True)
    assert anchored.ratio(1, 2)["se_building_cluster"] == pytest.approx(exact, rel=1e-5)

    # The common offset should largely disappear from the numerator projection.
    j = evaluation.ClusterMoments.PAIRS.index((1, 2))
    assert abs(run(anchored=False).projected_numerator[j]) > 1e10 * abs(
        anchored.projected_numerator[j]
    )


def test_a_low_weight_anchor_preserves_cluster_variance():
    weights = np.array([1e-20, 1.0, 2.0, 3.0])
    rewards = np.array([0.0, 1e12 + 3, 1e12 - 7, 1e12 + 5])
    values = np.column_stack(
        [np.ones(4), weights * rewards, weights, rewards, rewards, np.zeros(4)]
    )
    moments = evaluation.ClusterMoments()
    moments.add(np.arange(4), values)
    moments.flush()

    weighted = [Fraction(v) for v in values[:, 1]]
    denominators = [Fraction(w) for w in weights]
    estimate = sum(weighted) / sum(denominators)
    residual_ss = sum((z - estimate * d) ** 2 for z, d in zip(weighted, denominators))
    exact = math.sqrt(float(residual_ss * Fraction(4, 3) / sum(denominators) ** 2))
    assert moments.ratio(1, 2)["se_building_cluster"] == pytest.approx(exact, rel=5e-5)


def test_small_cluster_does_not_distort_ratio_for_dominant_clusters():
    moments = evaluation.ClusterMoments()
    moments.add(
        np.arange(3),
        np.array(
            [
                [1.0, -1e30, 1.0, -1e30, -1e30, 0.0],
                [1.0, -1e30, 1e30, -1.0, -1.0, 0.0],
                [1.0, -1e30, 1e30, -1.0, -1.0, 0.0],
            ]
        ),
    )
    moments.flush()
    result = moments.ratio(1, 2)
    assert result["value"] == pytest.approx(-1.5)
    assert result["se_building_cluster"] == pytest.approx(0.75)


@pytest.mark.parametrize("buildings", [(0, 0, 0), (0, 1, 2)])
@pytest.mark.parametrize("cut", [0, 1, 2])
def test_signed_cancellation_preserves_small_rewards_across_batch_boundaries(buildings, cut):
    ids = np.array(buildings)
    rewards = np.array([1e16, 1.0, -1e16])
    values = np.column_stack([np.ones(3), rewards, np.ones(3), rewards, rewards, np.zeros(3)])
    moments = evaluation.ClusterMoments()
    moments.add(ids[:cut], values[:cut])
    moments.add(ids[cut:], values[cut:])
    moments.flush()
    for numerator, denominator in ((1, 0), (1, 2), (3, 0), (4, 0)):
        assert moments.ratio(numerator, denominator)["value"] == pytest.approx(1 / 3)


def test_cluster_order_and_empty_input():
    m = evaluation.ClusterMoments()
    m.add(np.array([], dtype=int), np.empty((0, 6)))
    with pytest.raises(ValueError, match="ordered"):
        m.add(np.array([2, 1]), np.ones((2, 6)))
    m.add(np.array([1]), np.ones((1, 6)))
    m.flush()
    with pytest.raises(ValueError, match="reopen"):
        m.add(np.array([1]), np.ones((1, 6)))


@pytest.mark.parametrize(
    "values",
    [
        np.full((1, 6), 1 + 2j),
        np.full((1, 6), np.complex128(1 + 2j), dtype=object),
        np.full((1, 6), 10**400, dtype=object),
    ],
)
def test_cluster_rejects_values_outside_real_float64_before_updating(values):
    moments = evaluation.ClusterMoments()
    moments.add(np.array([0]), np.ones((1, 6)))
    with pytest.raises(ValueError, match="representable as float64"):
        moments.add(np.array([1]), values)
    moments.flush()
    assert moments.count == 1
    assert moments.ratio(1, 0)["value"] == 1.0


def test_flushing_without_a_pending_building_is_a_no_op():
    m = evaluation.ClusterMoments()
    m.flush()
    assert m.count == 0 and m.ratio(1, 0)["value"] is None
    m.add(np.array([4]), np.ones((1, 6)))
    m.flush()
    m.flush()
    assert m.count == 1 and m.ratio(1, 0)["value"] == 1.0


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


@pytest.mark.parametrize("propensity", [1e-200, np.finfo(float).smallest_subnormal])
def test_policy_handles_very_small_positive_propensities(monkeypatch, policy_tables, propensity):
    logged, oracle = policy_tables
    gap = logged["obs_temp_last_c"].to_numpy().astype(float) - logged[
        "setpoint_c"
    ].to_numpy().astype(float)
    actions = np.where(gap > 1.5, 2, np.where(gap > -0.3, 1, 0))
    first = np.flatnonzero(actions[:24] == logged["action"].to_numpy()[:24])[0]
    logged, oracle = logged.take([first, 24]), oracle.take([first, 24])
    selected = logged["action"][0].as_py()
    for column in ("propensity", "p_action_0", "p_action_1", "p_action_2"):
        value = propensity if column in ("propensity", f"p_action_{selected}") else 0.5
        field = logged.schema.field(column)
        logged = logged.set_column(
            logged.schema.get_field_index(column),
            field,
            pa.array([value, logged[column][1].as_py()], type=field.type),
        )

    def batches(*args, **kwargs):
        yield {}, logged, oracle

    monkeypatch.setattr(evaluation, "iter_pairs", batches)
    if propensity == np.finfo(float).smallest_subnormal:
        with pytest.raises(ValueError, match="exceeds the float64 range"):
            evaluation.evaluate_policy("unused")
        return

    estimate = logged["y_reward"][0].as_py() / (2 * propensity)
    result = evaluation.evaluate_policy("unused")
    assert result["ips"]["value"] == pytest.approx(estimate)
    assert result["ips"]["se_building_cluster"] == pytest.approx(abs(estimate))
    assert result["row_weight_ess_diagnostic"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "propensity,reward",
    [
        (1e-308, -1.0),
        (np.finfo(float).smallest_subnormal, -np.finfo(np.float32).smallest_subnormal),
    ],
)
def test_finite_policy_ratios_survive_overflow_in_unscaled_totals(
    monkeypatch, policy_tables, propensity, reward
):
    logged, oracle = (table.take([0, 24]) for table in policy_tables)
    replacements = {
        "obs_temp_last_c": logged["setpoint_c"],
        "action": 1,
        "propensity": propensity,
        "p_action_0": 0.5,
        "p_action_1": propensity,
        "p_action_2": 0.5,
        "y_reward": reward,
    }
    for column, value in replacements.items():
        field = logged.schema.field(column)
        data = (
            value if isinstance(value, pa.ChunkedArray) else pa.array([value] * 2, type=field.type)
        )
        logged = logged.set_column(logged.schema.get_field_index(column), field, data)
    oracle = oracle.set_column(
        oracle.schema.get_field_index("cf_reward_1"),
        oracle.schema.field("cf_reward_1"),
        pa.array([reward] * 2, type=pa.float32()),
    )

    def batches(*args, **kwargs):
        yield {}, logged, oracle

    monkeypatch.setattr(evaluation, "iter_pairs", batches)
    result = evaluation.evaluate_policy("unused")
    assert result["ips"]["value"] == pytest.approx(float(reward) / propensity)
    assert result["ips"]["se_building_cluster"] == 0.0
    assert result["snips"]["value"] == pytest.approx(float(reward), rel=1e-12, abs=0.0)
    assert result["row_weight_ess_diagnostic"] == pytest.approx(2.0)


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


def test_logging_probabilities_that_do_not_sum_to_one_are_rejected(monkeypatch, policy_tables):
    # Scaling all three probabilities and the propensity together keeps every other
    # guard satisfied -- each value stays inside (0, 1] and the selected propensity
    # still matches its column -- so only the simplex check can fire. A release read
    # here has not necessarily been through validate().
    logged, oracle = policy_tables
    for column in ("p_action_0", "p_action_1", "p_action_2", "propensity"):
        field = logged.schema.field(column)
        scaled = [value * 0.9 for value in logged[column].to_pylist()]
        logged = logged.set_column(
            logged.schema.get_field_index(column), field, pa.array(scaled, type=field.type)
        )
    probabilities = np.column_stack([logged[f"p_action_{a}"].to_numpy() for a in range(3)])
    assert (probabilities > 0).all() and (probabilities <= 1).all()

    def batches(*args, **kwargs):
        yield {}, logged, oracle

    monkeypatch.setattr(evaluation, "iter_pairs", batches)
    with pytest.raises(ValueError, match="logging probabilities do not sum to 1"):
        evaluation.evaluate_policy("unused")


@pytest.mark.parametrize("split", [None, "unknown", 1, []])
def test_policy_rejects_invalid_split_before_reading(monkeypatch, split):
    def unexpected_read(*args, **kwargs):
        raise AssertionError("invalid split reached the reader")

    monkeypatch.setattr(evaluation, "iter_pairs", unexpected_read)
    with pytest.raises(ValueError, match="unknown split"):
        evaluation.evaluate_policy("unused", split=split)
