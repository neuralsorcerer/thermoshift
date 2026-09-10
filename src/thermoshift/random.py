# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stateless draws indexed by building, stream and hour using SplitMix64."""

import numpy as np

MASK = (1 << 64) - 1


def mix64(x):
    with np.errstate(over="ignore"):
        x = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def uniform(ids, seed, stream, step=0):
    key = (
        seed * 0xD2B74407B1CE6E93 + stream * 0xCA5A826395121157 + step * 0x9E3779B97F4A7C15
    ) & MASK
    bits = mix64(np.asarray(ids, dtype=np.uint64) ^ np.uint64(key))
    # Strictly inside (0,1), so log() in Box-Muller is safe.
    return ((bits >> np.uint64(12)).astype(np.float64) + 0.5) / 2**52


def normal(ids, seed, stream, step=0):
    u = uniform(ids, seed, 1000 + 2 * stream, step)
    v = uniform(ids, seed, 1001 + 2 * stream, step)
    return np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * np.pi * v)


def split_codes(ids, seed):
    # Independent of action/weather streams and total dataset size.
    bucket = mix64(np.asarray(ids, dtype=np.uint64) ^ np.uint64(seed)) % np.uint64(100)
    return np.searchsorted(np.array([70, 80, 90, 95, 100]), bucket, side="right").astype(np.int8)
