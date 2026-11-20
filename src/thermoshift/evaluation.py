# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""One-step policy evaluation with building-level cluster uncertainty."""

from __future__ import annotations

import math
from contextlib import closing
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np

from thermoshift._checks import array, require
from thermoshift.config import SPLITS
from thermoshift.reading import iter_pairs

# Leave headroom for the inverse of any positive float64 propensity, a float32
# reward, and up to 2**63 - 1 rows. This exact scaling cancels in SNIPS and ESS.
_POLICY_WEIGHT_SHIFT = 256


class ClusterMoments:
    """Centered cluster moments for five prespecified ratios in constant memory.

    Anchor each ratio and accumulate a two-column QR factor for its residual
    norm so large weights and offsets do not swamp small differences.
    """

    PAIRS = ((1, 0), (1, 2), (3, 0), (4, 0), (5, 0))

    def __init__(self):
        self.count = 0
        self.total = np.zeros(6)
        self.compensation = np.zeros(6)
        self.anchor = np.zeros(5)
        self.anchor_denominator = np.zeros(5)
        self.denominator_norm = np.zeros(5)
        self.projected_numerator = np.zeros(5)
        self.orthogonal_numerator_norm = np.zeros(5)
        self.centered_sum = np.zeros(5)
        self.centered_compensation = np.zeros(5)
        self.pending_id = None
        self.last_flushed_id = None
        self.pending = np.zeros(6)
        self.pending_compensation = np.zeros(6)

    @staticmethod
    def _accumulate(total, compensation, value):
        updated = total + value
        # Neumaier summation preserves the rounding error even when the
        # incoming term is larger than the running sum.
        total_is_larger = np.abs(total) >= np.abs(value)
        large = np.where(total_is_larger, total, value)
        small = np.where(total_is_larger, value, total)
        compensation += (large - updated) + small
        total[:] = updated

    def flush(self):
        if self.pending_id is None:
            return
        self.count += 1
        transformed = np.zeros((5, 2))
        for j, (num, den) in enumerate(self.PAIRS):
            z, d = self.pending[num], self.pending[den]
            z_error, d_error = self.pending_compensation[num], self.pending_compensation[den]
            total_denominator = d + d_error
            if total_denominator > self.anchor_denominator[j]:
                anchor = (z + z_error) / total_denominator
                shift = self.anchor[j] - anchor
                # Adding a multiple of the denominator changes the projection only.
                self.projected_numerator[j] += shift * self.denominator_norm[j]
                self._accumulate(
                    self.centered_sum[j : j + 1],
                    self.centered_compensation[j : j + 1],
                    shift * (self.total[den] + self.compensation[den]),
                )
                self.anchor[j] = anchor
                self.anchor_denominator[j] = total_denominator
            anchor = self.anchor[j]
            transformed[j] = (z - anchor * d) + (z_error - anchor * d_error), total_denominator
        numerator, denominator = transformed.T
        # A Givens rotation adds the building's row to the existing QR factor.
        norm = np.hypot(self.denominator_norm, denominator)
        active = norm != 0
        cosine = np.divide(self.denominator_norm, norm, out=np.ones_like(norm), where=active)
        sine = np.divide(denominator, norm, out=np.zeros_like(norm), where=active)
        self.orthogonal_numerator_norm = np.hypot(
            self.orthogonal_numerator_norm, cosine * numerator - sine * self.projected_numerator
        )
        self.projected_numerator = cosine * self.projected_numerator + sine * numerator
        self.denominator_norm = norm
        self._accumulate(self.centered_sum, self.centered_compensation, numerator)
        self._accumulate(self.total, self.compensation, self.pending)
        self._accumulate(self.total, self.compensation, self.pending_compensation)
        self.last_flushed_id = self.pending_id
        self.pending_id = None
        self.pending.fill(0)
        self.pending_compensation.fill(0)

    def add(self, ids, values):
        ids, values = np.asarray(ids), np.asarray(values)
        require(ids.ndim == 1 and values.shape == (len(ids), 6), "invalid cluster input shape")
        require(np.issubdtype(ids.dtype, np.integer), "building IDs must be integers")
        value_error = "cluster values must be real numbers representable as float64"
        require(
            not np.iscomplexobj(values)
            and not (
                values.dtype.kind == "O"
                and any(isinstance(value, (complex, np.complexfloating)) for value in values.flat)
            ),
            value_error,
        )
        try:
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                values = values.astype(float, copy=False)
        except (FloatingPointError, OverflowError, TypeError, ValueError) as error:
            raise ValueError(value_error) from error
        require(np.isfinite(values), "nonfinite cluster values")
        require((values[:, 0] > 0) & (values[:, 2] >= 0), "invalid count or importance weight")
        require(ids[1:] >= ids[:-1], "input must be ordered by building")
        if not len(ids):
            return
        require(ids[0] >= 0, "building IDs must be nonnegative")
        starts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1]
        ends = np.r_[starts[1:], len(ids)]
        for bid, start, end in zip(ids[starts], starts, ends):
            if self.pending_id is not None and bid != self.pending_id:
                require(bid > self.pending_id, "input must be ordered by building")
                self.flush()
            if self.pending_id is None and self.last_flushed_id is not None:
                require(bid > self.last_flushed_id, "cannot reopen a flushed building")
            self.pending_id = int(bid)
            if end - start == 1:
                self._accumulate(self.pending, self.pending_compensation, values[start])
                continue
            # Keep both parts of the subtotal when a building spans chunks.
            columns = values[start:end].T
            subtotal = np.array([math.fsum(column) for column in columns])
            rounding = np.array(
                [math.fsum(chain(column, (-total,))) for column, total in zip(columns, subtotal)]
            )
            self._accumulate(self.pending, self.pending_compensation, subtotal)
            self._accumulate(self.pending, self.pending_compensation, rounding)

    def ratio(self, numerator, denominator):
        return self._ratio(numerator, denominator)

    def _ratio(self, numerator, denominator, numerator_exponent=0):
        require(self.pending_id is None, "flush pending building before computing ratios")
        require((numerator, denominator) in self.PAIRS, "unsupported ratio")
        j = self.PAIRS.index((numerator, denominator))
        den = self.total[denominator] + self.compensation[denominator]
        if den <= 0:
            return {"value": None, "se_building_cluster": None, "ci95_normal_approx": None}
        correction = (self.centered_sum[j] + self.centered_compensation[j]) / den
        value = np.ldexp(
            (self.total[numerator] + self.compensation[numerator]) / den, numerator_exponent
        )
        if self.count <= 1 or (numerator == 1 and self.total[2] == 0):
            return {"value": float(value), "se_building_cluster": None, "ci95_normal_approx": None}
        residual_norm = np.hypot(
            self.projected_numerator[j] - correction * self.denominator_norm[j],
            self.orthogonal_numerator_norm[j],
        )
        se = np.ldexp(
            (residual_norm / den) * np.sqrt(self.count / (self.count - 1)), numerator_exponent
        )
        return {
            "value": float(value),
            "se_building_cluster": float(se),
            "ci95_normal_approx": [float(value - 1.96 * se), float(value + 1.96 * se)],
        }


def evaluate_policy(root: str | Path, split: str = "test") -> dict[str, Any]:
    """Evaluate the fixed threshold policy on logged states using IPS, SNIPS and oracle outcomes."""
    require(isinstance(split, str) and split in SPLITS, "unknown split")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
            return _evaluate_policy(root, split)
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("policy evaluation exceeds the float64 range") from error


def _evaluate_policy(root, split):
    moments = ClusterMoments()
    weight_norm = 0.0
    row_count = 0
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
            weight = (action == logged_action) * math.ldexp(1.0, -_POLICY_WEIGHT_SHIFT) / propensity
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
            weight_norm = np.hypot(weight_norm, np.hypot.reduce(weight, initial=0.0))
            row_count += x.num_rows
    moments.flush()
    require(row_count > 0, "no rows found for this split")
    return {
        "split": split,
        "rows": row_count,
        "independent_buildings": moments.count,
        "estimand": "one-step reward on the logging-policy state distribution",
        "overlap_status": "observed_matches" if weight_norm else "no_observed_matches",
        "ips": moments._ratio(1, 0, _POLICY_WEIGHT_SHIFT),
        "snips": moments.ratio(1, 2),
        "oracle_policy_value": moments.ratio(3, 0),
        "logged_policy_value": moments.ratio(4, 0),
        "oracle_one_step_regret": moments.ratio(5, 0),
        "row_weight_ess_diagnostic": float(
            ((moments.total[2] + moments.compensation[2]) / weight_norm) ** 2
        )
        if weight_norm
        else 0,
        "notes": [
            "Normal intervals are asymptotic over independent buildings; few-building intervals may be unreliable.",
            "No observed matches means IPS is zero but its interval is suppressed; SNIPS is undefined.",
            "Use the supplied true logging propensities; re-estimation from observed features in hidden-confounding mode is not generally valid.",
            "Row-weight ESS is an overlap diagnostic, not an independent-sample count.",
        ],
    }
