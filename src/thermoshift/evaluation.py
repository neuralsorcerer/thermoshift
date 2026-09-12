# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""One-step policy evaluation with building-level cluster uncertainty."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

from thermoshift._checks import array, require
from thermoshift.config import SPLITS
from thermoshift.reading import iter_pairs


class ClusterMoments:
    """Centered cluster moments for five prespecified ratios in constant memory.

    Subtract a fixed ratio anchor BEFORE taking second moments to avoid
    catastrophic cancellation for outcomes with a large common offset.
    """

    PAIRS = ((1, 0), (1, 2), (3, 0), (4, 0), (5, 0))

    def __init__(self):
        self.count = 0
        self.total = np.zeros(6)
        self.compensation = np.zeros(6)
        self.anchor = np.full(5, np.nan)
        self.mean = np.zeros((5, 2))
        self.m2 = np.zeros((5, 2, 2))
        self.centered_sum = np.zeros((5, 2))
        self.centered_compensation = np.zeros((5, 2))
        self.pending_id = None
        self.last_flushed_id = None
        self.pending = np.zeros(6)

    @staticmethod
    def _accumulate(total, compensation, value):
        adjusted = value - compensation
        updated = total + adjusted
        compensation[:] = (updated - total) - adjusted
        total[:] = updated

    def flush(self):
        if self.pending_id is None:
            return
        self.count += 1
        self._accumulate(self.total, self.compensation, self.pending)
        transformed = np.zeros((5, 2))
        for j, (num, den) in enumerate(self.PAIRS):
            z, d = self.pending[num], self.pending[den]
            if np.isnan(self.anchor[j]) and d != 0:
                self.anchor[j] = z / d
            anchor = 0 if np.isnan(self.anchor[j]) else self.anchor[j]
            transformed[j] = z - anchor * d, d
        delta = transformed - self.mean
        self.mean += delta / self.count
        self.m2 += delta[:, :, None] * (transformed - self.mean)[:, None, :]
        self._accumulate(self.centered_sum, self.centered_compensation, transformed)
        self.last_flushed_id = self.pending_id
        self.pending_id, self.pending = None, np.zeros(6)

    def add(self, ids, values):
        ids, values = np.asarray(ids), np.asarray(values, dtype=float)
        require(ids.ndim == 1 and values.shape == (len(ids), 6), "invalid cluster input shape")
        require(np.issubdtype(ids.dtype, np.integer), "building IDs must be integers")
        require(np.isfinite(values), "nonfinite cluster values")
        require((values[:, 0] > 0) & (values[:, 2] >= 0), "invalid count or importance weight")
        require(ids[1:] >= ids[:-1], "input must be ordered by building")
        if not len(ids):
            return
        require(ids[0] >= 0, "building IDs must be nonnegative")
        starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
        sums = np.add.reduceat(values, starts, axis=0)
        for bid, subtotal in zip(ids[starts], sums):
            if self.pending_id is not None and bid != self.pending_id:
                require(bid > self.pending_id, "input must be ordered by building")
                self.flush()
            if self.pending_id is None and self.last_flushed_id is not None:
                require(bid > self.last_flushed_id, "cannot reopen a flushed building")
            self.pending_id = int(bid)
            self.pending += subtotal

    def ratio(self, numerator, denominator):
        require(self.pending_id is None, "flush pending building before computing ratios")
        require((numerator, denominator) in self.PAIRS, "unsupported ratio")
        j = self.PAIRS.index((numerator, denominator))
        den = self.centered_sum[j, 1]
        if den <= 0:
            return {"value": None, "se_building_cluster": None, "ci95_normal_approx": None}
        correction = self.centered_sum[j, 0] / den
        anchor = 0 if np.isnan(self.anchor[j]) else self.anchor[j]
        value = anchor + correction
        if self.count <= 1 or (numerator == 1 and self.total[2] == 0):
            return {"value": float(value), "se_building_cluster": None, "ci95_normal_approx": None}
        v = np.array([1.0, -correction])
        residual_ss = float(v @ self.m2[j] @ v + self.count * (self.mean[j] @ v) ** 2)
        se = np.sqrt(max(0, residual_ss) * self.count / (self.count - 1)) / den
        return {
            "value": float(value),
            "se_building_cluster": float(se),
            "ci95_normal_approx": [float(value - 1.96 * se), float(value + 1.96 * se)],
        }


def evaluate_policy(root: str | Path, split: str = "test") -> dict[str, Any]:
    """Evaluate the fixed threshold policy on logged states using IPS, SNIPS and oracle outcomes."""
    require(isinstance(split, str) and split in SPLITS, "unknown split")
    moments = ClusterMoments()
    weight_squared = 0.0
    with closing(iter_pairs(root, split=split)) as batches:
        for _, x, o in batches:
            require(x.num_rows == o.num_rows, "oracle alignment fails")
            for name in ("row_id", "building_id", "step"):
                require(array(x, name) == array(o, name), "oracle alignment fails")
            temperature = array(x, "obs_temp_last_c").astype(float)
            setpoint = array(x, "setpoint_c").astype(float)
            require(np.isfinite(temperature) & np.isfinite(setpoint), "nonfinite policy input")
            gap = temperature - setpoint
            action = np.where(gap > 1.5, 2, np.where(gap > -0.3, 1, 0))
            logged_action = array(x, "action")
            require((logged_action >= 0) & (logged_action < 3), "invalid logged action")
            propensity = array(x, "propensity").astype(float)
            require(
                np.isfinite(propensity) & (propensity > 0) & (propensity <= 1),
                "invalid logging propensity",
            )
            probabilities = np.column_stack([array(x, f"p_action_{a}") for a in range(3)])
            require(
                np.isfinite(probabilities) & (probabilities > 0) & (probabilities <= 1),
                "invalid logging probabilities",
            )
            require(
                np.isclose(probabilities.sum(axis=1), 1, atol=2e-7, rtol=2e-6),
                "logging probabilities do not sum to 1",
            )
            rows = np.arange(len(action))
            require(
                propensity == probabilities[rows, logged_action],
                "selected propensity mismatch",
            )
            weight = (action == logged_action) / propensity
            true_rewards = np.column_stack([array(o, f"cf_reward_{a}") for a in range(3)]).astype(
                float
            )
            require(np.isfinite(true_rewards), "nonfinite oracle reward")
            truth = true_rewards[rows, action]
            factual = array(x, "y_reward").astype(float)
            require(factual == true_rewards[rows, logged_action], "factual reward mismatch")
            regret = true_rewards.max(axis=1) - truth
            # count, weighted reward, weight, oracle reward, factual reward, regret
            values = np.column_stack(
                [np.ones(len(action)), weight * factual, weight, truth, factual, regret]
            )
            moments.add(array(x, "building_id"), values)
            weight_squared += float(np.dot(weight, weight))
    moments.flush()
    require(moments.total[0] > 0, "no rows found for this split")
    return {
        "split": split,
        "rows": int(moments.total[0]),
        "independent_buildings": moments.count,
        "estimand": "one-step reward on the logging-policy state distribution",
        "overlap_status": "observed_matches" if weight_squared else "no_observed_matches",
        "ips": moments.ratio(1, 0),
        "snips": moments.ratio(1, 2),
        "oracle_policy_value": moments.ratio(3, 0),
        "logged_policy_value": moments.ratio(4, 0),
        "oracle_one_step_regret": moments.ratio(5, 0),
        "row_weight_ess_diagnostic": float(moments.total[2] ** 2 / weight_squared)
        if weight_squared
        else 0,
        "notes": [
            "Normal intervals are asymptotic over independent buildings; few-building intervals may be unreliable.",
            "No observed matches means IPS is zero but its interval is suppressed; SNIPS is undefined.",
            "Use the supplied true logging propensities; re-estimation from observed features in hidden-confounding mode is not generally valid.",
            "Row-weight ESS is an overlap diagnostic, not an independent-sample count.",
        ],
    }
