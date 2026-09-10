# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Array conversion and validation assertions shared by streaming consumers."""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def array(table: pa.Table, name: str) -> np.ndarray:
    return table[name].to_numpy(zero_copy_only=False)


def require(condition, message):
    if not bool(np.all(condition)):
        raise ValueError(message)


def close(actual, expected, message, atol=1e-4):
    require(np.isclose(actual, expected, atol=atol, rtol=2e-6), message)
