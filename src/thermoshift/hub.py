# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Publish validated datasets with provenance binding and remote checksum checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from thermoshift.config import boolean
from thermoshift.filesystem import atomic_json, dataset_path, read_json, sha256
from thermoshift.lifecycle import dataset_lock

UPLOAD_PATTERNS = [
    "data/**/*.parquet",
    "README.md",
    "DATASHEET.md",
    "LICENSE",
    "run_config.json",
    "schema.json",
    "feature_roles.json",
    "manifest.json",
    "validation.json",
    "_PUBLICATION.json",
]


def _git_blob_sha1(path):
    path = Path(path)
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_remote(api, root, repo_id, commit_sha, required):
    """Verify the immutable uploaded tree, including files skipped by ignore rules."""
    from huggingface_hub import RepoFile

    remote = {
        f.path: f
        for f in api.list_repo_tree(
            repo_id=repo_id, repo_type="dataset", revision=commit_sha, recursive=True
        )
        if isinstance(f, RepoFile)
    }
    if any(name.startswith("data/") and name not in required for name in remote):
        raise ValueError("uploaded tree contains additional data files")
    for name in required:
        local, item = Path(root) / name, remote.get(name)
        if item is None or item.size != local.stat().st_size:
            raise ValueError(
                f"remote payload missing or wrong size: {name}; check upload ignore rules"
            )
        if item.lfs is not None:
            matches = item.lfs.sha256 == sha256(local) and item.lfs.size == item.size
        else:
            matches = item.blob_id == _git_blob_sha1(local)
        if not matches:
            raise ValueError(f"remote payload checksum mismatch: {name}")


def publish(
    root: str | Path,
    repo_id: str,
    public: bool = False,
    revision: str = "main",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate a stable snapshot; mark complete only after remote verification.

    upload_folder may split into commits. Its parent guard protects the first
    commit only; we verify the final immutable tree and guard the marker commit.
    A local and remote intent binds retries to this exact release and target.
    """
    boolean(public, "public")
    boolean(dry_run, "dry_run")
    try:
        from huggingface_hub.utils import validate_repo_id
    except ImportError as error:
        raise ValueError(
            "Publishing requires the hub extra: python -m pip install '.[hub]'"
        ) from error

    from thermoshift.reading import _validated_manifest
    from thermoshift.validation import _validate_locked

    validate_repo_id(repo_id)
    if repo_id.count("/") != 1:
        raise ValueError("repo ID must be OWNER/DATASET")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("revision must name an existing branch")
    root = Path(root).resolve()
    with dataset_lock(root):
        # Never rely on an old cached validation report for a publication gate.
        _validate_locked(root, full=True)
        config, manifest = _validated_manifest(root, verify_hash=False)
        success = read_json(root / "_SUCCESS.json")
        intent = {
            "manifest_sha256": success["manifest_sha256"],
            "metadata_sha256": success["metadata_sha256"],
        }
        plan = {
            "repo_id": repo_id,
            "revision": revision,
            "visibility": "public" if public else "private",
            "decision_rows": config.rows,
            "parquet_files": len(manifest["files"]),
            "parquet_bytes": manifest["bytes"],
            "allow_patterns": list(UPLOAD_PATTERNS),
            "upload_method": "HfApi.upload_folder",
            "dry_run": dry_run,
            "steps": [
                "full local validation",
                "guarded provenance and intent commit",
                "payload upload",
                "verify immutable remote sizes and hashes",
                "guarded _SUCCESS.json commit",
            ],
        }
        if dry_run:
            return plan
        from huggingface_hub import (
            CommitOperationAdd,
            CommitOperationDelete,
            HfApi,
            hf_hub_download,
        )
        from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

        binding_path = dataset_path(root, "_state/hub_binding.json")
        publication_path = dataset_path(root, "_PUBLICATION.json")
        binding = {"repo_id": repo_id, "revision": revision, "public": public, **intent}
        if binding_path.exists() and read_json(binding_path) != binding:
            raise ValueError("this output is already bound to a different Hub target or release")
        api = HfApi()  # hf auth login or HF_TOKEN; no credentials enter release files.
        try:
            info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
        except RevisionNotFoundError:
            raise ValueError("revision does not exist; create the branch first") from None
        except RepositoryNotFoundError:
            if revision != "main":
                raise ValueError(
                    "new repositories require revision main; create another branch separately"
                ) from None
            api.create_repo(
                repo_id=repo_id, repo_type="dataset", private=not public, exist_ok=False
            )
            info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=revision)
        if info.private != (not public):
            raise ValueError(
                "existing repository visibility differs from the command; use the matching flag or a new repository"
            )
        head = info.sha
        if not isinstance(head, str) or not head:
            raise ValueError("Hub did not return an immutable repository head SHA")
        remote_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=head)
        if any(f != ".gitattributes" for f in remote_files):
            if not {"run_config.json", "_PUBLICATION.json"}.issubset(remote_files):
                raise ValueError(
                    "remote branch contains files but no matching ThermoShift publication intent"
                )
            for name, expected in (
                ("run_config.json", read_json(root / "run_config.json")),
                ("_PUBLICATION.json", intent),
            ):
                cached = hf_hub_download(repo_id, name, repo_type="dataset", revision=head)
                if read_json(cached) != expected:
                    raise ValueError(
                        "remote release has different provenance or content; use a new repository or branch"
                    )
        expected_data = {f["path"] for f in manifest["files"]}
        if any(f.startswith("data/") and f not in expected_data for f in remote_files):
            raise ValueError(
                "remote contains additional data files; use a fresh release repository"
            )
        atomic_json(publication_path, intent)
        atomic_json(binding_path, binding)
        operations = [
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=root / name)
            for name in ("run_config.json", "_PUBLICATION.json")
        ]
        # A retry can overwrite a previously completed tree. Invalidate its
        # marker in the guarded commit before uploading any replacement bytes.
        if "_SUCCESS.json" in remote_files:
            operations.append(CommitOperationDelete(path_in_repo="_SUCCESS.json"))
        provenance = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            parent_commit=head,
            operations=operations,
            commit_message="Record ThermoShift provenance and immutable publication intent",
        )
        payload = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=root,
            revision=revision,
            parent_commit=provenance.oid,
            allow_patterns=UPLOAD_PATTERNS,
            commit_message=f"Publish ThermoShift {config.rows:,} decisions ({config.fingerprint[:12]})",
        )
        required = expected_data | {name for name in UPLOAD_PATTERNS if "*" not in name}
        _verify_remote(api, root, repo_id, payload.oid, required)
        commit = api.upload_file(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            parent_commit=payload.oid,
            path_or_fileobj=root / "_SUCCESS.json",
            path_in_repo="_SUCCESS.json",
            commit_message="Mark the verified complete ThermoShift release",
        )
        # Verify marker bytes as well; return a pinned revision for reproducible fetches.
        _verify_remote(api, root, repo_id, commit.oid, required | {"_SUCCESS.json"})
        return {
            **plan,
            "commit_sha": commit.oid,
            "url": f"https://huggingface.co/datasets/{repo_id}/tree/{commit.oid}",
        }
