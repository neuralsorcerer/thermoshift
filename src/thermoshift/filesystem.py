# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Checksums and atomic local-file operations."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import tempfile
from itertools import chain
from pathlib import Path
from typing import Any

# Windows exposes the junction tag through lstat even on Python 3.11, which
# predates os.path.isjunction.
_JUNCTION_TAG = getattr(stat, "IO_REPARSE_TAG_MOUNT_POINT", None)


def is_redirected(path: Path) -> bool:
    """Report a symlink, or on Windows a junction, at this exact path."""
    try:
        info = path.lstat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    return stat.S_ISLNK(info.st_mode) or (
        _JUNCTION_TAG is not None and getattr(info, "st_reparse_tag", None) == _JUNCTION_TAG
    )


def dataset_path(root: str | Path, relative: str | Path) -> Path:
    """Resolve an internal dataset path without following redirected components.

    The dataset root itself may be a caller-selected symlink. Paths within it
    must stay local so generation cannot write through a redirected partition,
    temporary file or lock file. Lifecycle locks exclude cooperating writers;
    unrelated programs must still leave active dataset paths unchanged.
    """
    relative = Path(relative)
    # Reject on the anchor rather than is_absolute(): on Windows "/data" carries
    # no drive and is not absolute, yet joining it resets to the drive root and
    # leaves the release entirely. "C:data" is drive-relative the same way.
    if relative.anchor or any(part == ".." or Path(part).anchor for part in relative.parts):
        raise ValueError("dataset paths must be relative to the dataset directory")
    path = Path(root)
    for part in relative.parts:
        path /= part
        if is_redirected(path):
            raise ValueError(f"symlinks and junctions are not allowed in dataset paths: {relative}")
    return path


def dataset_parquet_paths(root: Path) -> set[str]:
    """List Parquet paths in the data tree, failing on redirects or unreadable entries."""
    data = dataset_path(root, "data")
    if not data.exists():
        return set()

    def raise_scan_error(error):
        raise error

    files = set()
    for parent, directories, names in os.walk(data, onerror=raise_scan_error):
        for name in chain(directories, names):
            path = Path(parent, name)
            if is_redirected(path):
                raise ValueError("symlinks and junctions are not allowed under data/")
            if path.match("*.parquet"):
                files.add(path.relative_to(root).as_posix())
    return files


def require_regular_file(path: Path, *, missing_ok: bool = False) -> None:
    """Reject directories and special files before opening a dataset file."""
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        if missing_ok:
            return
        raise
    if not stat.S_ISREG(mode):
        raise ValueError(f"expected a regular file at {path}")


def sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    path = Path(path)
    require_regular_file(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fsync_directory(path: str | Path) -> None:
    """Flush a directory entry when the filesystem supports directory fsync."""
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            try:
                os.fsync(fd)
            except OSError as error:
                if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                    raise
        finally:
            os.close(fd)


def atomic_bytes(path: str | Path, data: bytes) -> None:
    """Write and flush bytes, then replace the destination with an atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: str | Path, value: Any) -> None:
    """Atomically write UTF-8 JSON with stable keys and finite numeric values."""
    atomic_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document."""
    path = Path(path)
    require_regular_file(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except RecursionError as error:
        raise ValueError(f"JSON nesting exceeds the parser limit in {path}") from error
