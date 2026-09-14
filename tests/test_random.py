# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The indexed draws every release is reproduced from.

These streams are the reproducibility contract: the same seed must rebuild the
same dataset on any machine, forever. The references below are transcribed from
the published SplitMix64 algorithm rather than captured from this implementation,
so they test that the mixer *is* SplitMix64, not merely that it has not changed.
"""

import math

import numpy as np
import pytest

from thermoshift.random import mix64, normal, split_codes, uniform

MASK = (1 << 64) - 1

# The streams the simulator actually draws, so the independence check covers what
# releases are really built from rather than an arbitrary range.
UNIFORM_STREAMS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 102, 103, 104, 107, 109)
NORMAL_STREAMS = (11, 12, 13, 14, 101, 105, 106, 108, 110)
SEEDS = (0, 1, 42, 7919, 123456, 2**31 + 5)


def reference_mix64(x):
    """SplitMix64's finalizer, from the published algorithm, in exact integers."""
    z = (x + 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def reference_uniform(i, seed, stream, step=0):
    key = (
        seed * 0xD2B74407B1CE6E93 + stream * 0xCA5A826395121157 + step * 0x9E3779B97F4A7C15
    ) & MASK
    return ((reference_mix64(i ^ key) >> 12) + 0.5) / 2**52


def reference_normal(i, seed, stream, step=0):
    u = reference_uniform(i, seed, 1000 + 2 * stream, step)
    v = reference_uniform(i, seed, 1001 + 2 * stream, step)
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def reference_split_codes(i, seed):
    # The source lists a trailing 100 boundary, which no bucket can reach because
    # the modulus caps them at 99; it is there so the array reads as the
    # cumulative allocation. Four boundaries decide every reachable bucket.
    bucket = reference_mix64(i ^ seed) % 100
    return sum(1 for boundary in (70, 80, 90, 95) if bucket >= boundary)


def test_mix64_is_the_published_splitmix64_finalizer():
    # Probe the ends of the word as well as ordinary values: the unsigned
    # wraparound in the two multiplies is the part a rewrite is most likely to
    # get wrong, and it only shows up near the top of the range.
    probe = [0, 1, 2, 3, 7, 255, 4096, 65535, 10**6, 2**31, 2**32, 2**52, 2**63, MASK - 1, MASK]
    got = mix64(np.array(probe, dtype=np.uint64))
    assert [int(v) for v in got] == [reference_mix64(v) for v in probe]


def test_uniform_matches_the_reference_across_seeds_streams_and_steps():
    # Every arithmetic step here is exact in float64 -- the 52-bit integer, the
    # half offset and the power-of-two divisor all are -- so this compares equal,
    # not merely close, and pins the key derivation along with it.
    ids = np.arange(200)
    for seed in SEEDS:
        for stream in (0, 1, 9, 110, 1000, 123456):
            for step in (0, 1, 23, 8759):
                got = uniform(ids, seed, stream, step)
                want = [reference_uniform(int(i), seed, stream, step) for i in ids]
                assert list(got) == want


def test_uniform_stays_strictly_inside_the_open_unit_interval():
    # log() in Box-Muller needs a value that is never exactly zero. The smallest
    # and largest representable draws are what the half offset buys, so check
    # them directly rather than hoping a sample lands there.
    smallest = (0 + 0.5) / 2**52
    largest = ((2**52 - 1) + 0.5) / 2**52
    assert 0 < smallest and largest < 1
    for seed in SEEDS:
        for stream in UNIFORM_STREAMS:
            values = uniform(np.arange(4000), seed, stream, 7)
            assert values.min() >= smallest and values.max() <= largest


def test_normal_is_box_muller_over_the_two_reserved_uniform_streams():
    # The underlying uniforms compare exactly; only sqrt, log and cos differ from
    # the pure-Python reference, by at most one ULP here. 1e-12 leaves several
    # thousand ULPs of room for a less accurate libm while still being far below
    # the O(1) change any mistake in the pairing would make.
    ids = np.arange(500)
    for seed in SEEDS:
        for stream in NORMAL_STREAMS:
            got = normal(ids, seed, stream, 5)
            want = [reference_normal(int(i), seed, stream, 5) for i in ids]
            assert got == pytest.approx(want, rel=1e-12)


def test_streams_the_simulator_draws_stay_independent():
    # A key that ignored the stream would make every building parameter the same
    # draw -- correlation 1.0 -- while leaving every documented range intact.
    # Correlation is the wrong instrument for subtler pairing mistakes, such as
    # normal() reserving its two uniform streams one apart instead of two; the
    # reference test above pins which stream feeds which draw.
    ids = np.arange(6000)
    for seed in SEEDS:
        draws = np.array([uniform(ids, seed, s, 3) for s in UNIFORM_STREAMS])
        draws = np.vstack([draws, [normal(ids, seed, s, 3) for s in NORMAL_STREAMS]])
        assert len(np.unique(draws, axis=0)) == len(draws), "two streams produced the same draw"
        off_diagonal = np.abs(np.corrcoef(draws) - np.eye(len(draws)))
        # Measured worst over these seeds is 0.046 at n=6000, against 1/sqrt(n)=0.013.
        assert off_diagonal.max() < 0.10


def test_consecutive_steps_of_one_stream_stay_independent():
    # The step term carries the hour, so a key that dropped it would repeat one
    # draw for every hour of a trajectory.
    ids = np.arange(6000)
    for seed in (0, 42, 123456):
        hours = np.array([uniform(ids, seed, 102, t) for t in range(12)])
        assert len(np.unique(hours, axis=0)) == len(hours)
        assert np.abs(np.corrcoef(hours) - np.eye(len(hours))).max() < 0.10


def test_split_codes_match_the_reference_derivation():
    # Proportions and position independence both survive a changed derivation:
    # combining the id with the seed by addition instead of exclusive-or keeps
    # 70/10/10/5/5 intact while reassigning every building in every release.
    ids = np.arange(500)
    for seed in SEEDS:
        got = split_codes(ids, seed)
        assert [int(v) for v in got] == [reference_split_codes(int(i), seed) for i in ids]


def test_split_codes_hold_the_documented_proportions():
    # README documents 70/10/10/5/5. An off-by-one at the bucket boundaries shifts
    # the allocation by a whole point while leaving the splits non-empty and
    # disjoint.
    ids = np.arange(200_000)
    documented = np.array([70.0, 10.0, 10.0, 5.0, 5.0])
    for seed in SEEDS:
        codes = split_codes(ids, seed)
        assert codes.dtype == np.int8
        assert set(np.unique(codes)) <= {0, 1, 2, 3, 4}
        percent = np.bincount(codes, minlength=5) / len(ids) * 100
        # Worst deviation measured across these seeds is 0.20 points.
        assert np.abs(percent - documented).max() < 0.5


def test_split_codes_are_position_independent_and_stable():
    # A building's split must not depend on how many buildings were generated
    # alongside it, or a larger release would reshuffle an earlier one's splits.
    ids = np.arange(5000)
    full = split_codes(ids, 42)
    np.testing.assert_array_equal(full[:1500], split_codes(ids[:1500], 42))
    np.testing.assert_array_equal(full[3000:], split_codes(ids[3000:], 42))
