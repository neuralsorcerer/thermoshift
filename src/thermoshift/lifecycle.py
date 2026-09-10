# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-process lifecycle exclusion for a local/shared-filesystem dataset.

Generators take shared leases, allowing disjoint shards to run concurrently.
Finalization, validation and publication take an exclusive lease. Every child
writer also takes a shared lease so a killed parent cannot expose active writes.
These are advisory locks: unrelated programs must not edit an active release.
"""

from contextlib import contextmanager
from pathlib import Path

import portalocker

from thermoshift.config import boolean


@contextmanager
def dataset_lock(root, *, generating=False):
    boolean(generating, "generating")
    root = Path(root)
    if not root.is_dir():
        raise ValueError("dataset directory does not exist; initialize it first")
    flags = (portalocker.LOCK_SH if generating else portalocker.LOCK_EX) | portalocker.LOCK_NB
    lock = portalocker.Lock(str(root / ".lifecycle.lock"), mode="a", flags=flags, timeout=0)
    try:
        lock.acquire()
    except portalocker.exceptions.LockException as error:
        raise ValueError(
            "dataset is busy: generation cannot overlap finalization, validation or publication"
        ) from error
    try:
        yield
    finally:
        lock.release()
