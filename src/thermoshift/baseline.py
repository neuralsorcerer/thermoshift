# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded temperature-regression training and held-out evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from thermoshift.config import SPLITS, integer
from thermoshift.filesystem import read_json
from thermoshift.lifecycle import dataset_lock
from thermoshift.reading import _validated_manifest, ordered_files
from thermoshift.schema import FEATURES


def read_bounded(root, split, limit):
    integer(limit, "limit", 1)
    if split not in SPLITS:
        raise ValueError("unknown split")
    with dataset_lock(root):
        _validated_manifest(root)
        return _read_bounded(root, split, limit)


def _read_bounded(root, split, limit):
    root = Path(root)
    columns = FEATURES + ["action", "y_next_temp_c"]
    chunks, count = [], 0
    for item in ordered_files(read_json(root / "manifest.json"), kind="logged", split=split):
        with pq.ParquetFile(root / item["path"]) as parquet:
            for batch in parquet.iter_batches(columns=columns, batch_size=8192):
                take = min(batch.num_rows, limit - count)
                chunks.append(pa.Table.from_batches([batch]).slice(0, take))
                count += take
                if count == limit:
                    return pa.concat_tables(chunks).to_pandas()
    return pa.concat_tables(chunks).to_pandas() if chunks else None


def matrix(frame):
    x = frame[FEATURES + ["action"]].copy()
    hour = x.pop("hour")
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return x.astype(np.float64)


def train_baseline(
    root: str | Path, max_train: int = 100_000, max_eval: int = 100_000
) -> dict[str, Any]:
    """Fit a temperature model on bounded training data and score disjoint evaluation splits."""
    integer(max_train, "max_train", 10)
    integer(max_eval, "max_eval", 1)
    with dataset_lock(root):
        _validated_manifest(root)
        return _run_locked(root, max_train, max_eval)


def _run_locked(root, max_train, max_eval):
    train = _read_bounded(root, "train", max_train)
    if train is None or len(train) < 10:
        raise ValueError("release has too few training records")
    model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=31, learning_rate=0.08, early_stopping=False, random_state=42
    )
    design = matrix(train)
    model.fit(design, train["y_next_temp_c"])
    results = {
        "training_rows": len(train),
        "features": list(design.columns),
        "oracle_used_for_training": False,
        "splits": {},
    }
    for split in SPLITS[1:]:
        frame = _read_bounded(root, split, max_eval)
        if frame is None:
            continue
        y = frame["y_next_temp_c"]
        pred = model.predict(matrix(frame))
        persistence = frame["obs_temp_last_c"]
        results["splits"][split] = {
            "rows": len(frame),
            "model_mae_c": float(mean_absolute_error(y, pred)),
            "model_rmse_c": float(root_mean_squared_error(y, pred)),
            "persistence_mae_c": float(mean_absolute_error(y, persistence)),
            "persistence_rmse_c": float(root_mean_squared_error(y, persistence)),
        }
    return results
