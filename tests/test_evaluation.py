# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from thermoshift import evaluation


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
