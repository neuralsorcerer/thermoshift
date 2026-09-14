# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded reading and the reference temperature baseline."""

import pytest

from thermoshift import Config
from thermoshift.baseline import matrix, read_bounded, train_baseline
from thermoshift.config import SPLITS
from thermoshift.filesystem import read_json
from thermoshift.schema import FEATURES, LOGGED_SCHEMA, feature_roles
from thermoshift.storage import generate, initialize


@pytest.fixture
def release(tmp_path):
    initialize(tmp_path, Config(rows=2400, episode_steps=24, shard_buildings=40))
    generate(tmp_path, batch_buildings=13, progress=None)
    return tmp_path


def test_bounded_reader_stops_at_the_limit_and_keeps_column_roles(release):
    frame = read_bounded(release, "train", 37)
    assert len(frame) == 37
    assert list(frame.columns) == FEATURES + ["action", "y_next_temp_c"]
    # A limit beyond the split returns the split, not an error.
    total = read_json(release / "manifest.json")["split_rows"]["train"]
    assert len(read_bounded(release, "train", total + 1000)) == total


def test_bounded_reader_reports_an_absent_split_as_none(tmp_path):
    initialize(tmp_path, Config(rows=10, episode_steps=5))
    generate(tmp_path, progress=None)
    empty = [s for s, n in read_json(tmp_path / "manifest.json")["split_rows"].items() if not n]
    assert empty, "fixture must leave at least one split empty"
    assert read_bounded(tmp_path, empty[0], 5) is None


@pytest.mark.parametrize("split,limit", [("nope", 5), ("train", 0), ("train", -1), ("train", True)])
def test_bounded_reader_rejects_invalid_arguments(release, split, limit):
    with pytest.raises(ValueError):
        read_bounded(release, split, limit)


def test_baseline_scores_every_nonempty_evaluation_split(release):
    result = train_baseline(release, max_train=1500, max_eval=400)
    assert result["oracle_used_for_training"] is False
    assert result["training_rows"] == 1500
    assert set(result["splits"]) <= set(SPLITS[1:])
    assert "train" not in result["splits"]
    for scores in result["splits"].values():
        assert 0 < scores["rows"] <= 400
        # RMSE dominates MAE for any error distribution.
        assert 0 <= scores["model_mae_c"] <= scores["model_rmse_c"]
        assert 0 <= scores["persistence_mae_c"] <= scores["persistence_rmse_c"]


def test_baseline_never_sees_targets_or_propensities(release):
    # A transition model may read the action; everything else outside the
    # declared feature roles is a target or logging detail and must stay out.
    features = train_baseline(release, max_train=600, max_eval=200)["features"]
    allowed = set(feature_roles()["transition_features"])
    forbidden = {f.name for f in LOGGED_SCHEMA} - allowed
    assert not forbidden.intersection(features)
    # The cyclic hour encoding replaces the raw hour.
    assert "hour" not in features
    assert {"hour_sin", "hour_cos", "action"} <= set(features)


def test_design_matrix_is_numeric_and_ordered(release):
    frame = read_bounded(release, "train", 50)
    design = matrix(frame)
    assert list(design.columns) == train_baseline(release, max_train=50, max_eval=50)["features"]
    assert all(str(dtype) == "float64" for dtype in design.dtypes)
    # Sensor dropout stays as NaN for the model's native missing-value handling.
    assert design["obs_temp_c"].isna().sum() == frame["obs_temp_c"].isna().sum()


def test_baseline_skips_evaluation_splits_with_no_files(tmp_path):
    initialize(tmp_path, Config(rows=15, episode_steps=3))
    generate(tmp_path, progress=None)
    counts = read_json(tmp_path / "manifest.json")["split_rows"]
    assert counts["train"] >= 10 and any(not counts[s] for s in SPLITS[1:])
    scored = train_baseline(tmp_path, max_train=10, max_eval=10)["splits"]
    assert set(scored) == {s for s in SPLITS[1:] if counts[s]}


def test_baseline_requires_enough_training_rows(tmp_path):
    initialize(tmp_path, Config(rows=12, episode_steps=4))
    generate(tmp_path, progress=None)
    assert read_json(tmp_path / "manifest.json")["split_rows"]["train"] < 10
    with pytest.raises(ValueError, match="too few training records"):
        train_baseline(tmp_path, max_train=10, max_eval=10)


@pytest.mark.parametrize("kwargs", [{"max_train": 9}, {"max_eval": 0}, {"max_train": 1.5}])
def test_baseline_rejects_invalid_limits(release, kwargs):
    with pytest.raises(ValueError):
        train_baseline(release, **kwargs)
