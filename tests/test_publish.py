# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stateful in-memory Hub tests. No real network publication is performed."""

import fnmatch
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import CommitOperationDelete, RepoFile
from huggingface_hub import HfApi as RealApi

from thermoshift import Config
from thermoshift.filesystem import atomic_json, read_json
from thermoshift.hub import publish
from thermoshift.storage import generate, initialize


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "release"
    initialize(root, Config(rows=1000, episode_steps=24))
    generate(root, progress=None)
    return root


class MemoryHub:
    """Enforce actual SDK signatures and parent-commit guards; retain snapshots."""

    def __init__(self, cache):
        self.cache = cache
        self.private = False
        self.head = "initial"
        self.trees = {self.head: {".gitattributes": b""}}
        self.calls = []
        self.skip = None
        self.corrupt = None
        self.fail_payload = False
        self.race_before_marker = False
        self.extra_after_payload = False

    def check(self, method, kwargs):
        inspect.signature(getattr(RealApi, method)).bind(self, **kwargs)
        self.calls.append((method, kwargs))

    def repo_info(self, **kw):
        self.check("repo_info", kw)
        return SimpleNamespace(private=self.private, sha=self.head)

    def list_repo_files(self, **kw):
        self.check("list_repo_files", kw)
        return list(self.trees[kw["revision"]])

    def commit(self, parent, additions, deletions=()):
        if parent != self.head:
            raise ValueError("mock parent commit conflict")
        tree = {**self.trees[self.head], **additions}
        for name in deletions:
            del tree[name]
        self.head = f"commit{len(self.trees):040d}"
        self.trees[self.head] = tree
        return SimpleNamespace(oid=self.head)

    def create_commit(self, **kw):
        self.check("create_commit", kw)
        return self.commit(
            kw["parent_commit"],
            {
                op.path_in_repo: Path(op.path_or_fileobj).read_bytes()
                for op in kw["operations"]
                if not isinstance(op, CommitOperationDelete)
            },
            [op.path_in_repo for op in kw["operations"] if isinstance(op, CommitOperationDelete)],
        )

    def upload_folder(self, **kw):
        self.check("upload_folder", kw)
        if self.fail_payload:
            raise OSError("mock interrupted network")
        root = Path(kw["folder_path"])
        additions = {}
        for path in root.rglob("*"):
            name = path.relative_to(root).as_posix()
            if (
                path.is_file()
                and name != self.skip
                and any(fnmatch.fnmatch(name, pat) for pat in kw["allow_patterns"])
            ):
                additions[name] = path.read_bytes()
                if name == self.corrupt:
                    additions[name] = b"X" * len(additions[name])
        if self.extra_after_payload:
            additions["data/unexpected.parquet"] = b"bad"
        result = self.commit(kw["parent_commit"], additions)
        if self.race_before_marker:
            self.commit(self.head, {"concurrent.txt": b"another writer"})
        return result

    def upload_file(self, **kw):
        self.check("upload_file", kw)
        return self.commit(
            kw["parent_commit"], {kw["path_in_repo"]: Path(kw["path_or_fileobj"]).read_bytes()}
        )

    def list_repo_tree(self, **kw):
        self.check("list_repo_tree", kw)
        result = []
        for name, data in self.trees[kw["revision"]].items():
            props = {
                "path": name,
                "size": len(data),
                "oid": hashlib.sha1(
                    f"blob {len(data)}\0".encode() + data, usedforsecurity=False
                ).hexdigest(),
            }
            if name.endswith(".parquet"):
                props["lfs"] = {
                    "size": len(data),
                    "oid": hashlib.sha256(data).hexdigest(),
                    "pointerSize": 132,
                }
            result.append(RepoFile(**props))
        return result

    def download(self, repo_id, filename, *, repo_type, revision):
        path = self.cache / revision / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.trees[revision][filename])
        return str(path)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    import huggingface_hub

    service = MemoryHub(tmp_path / "cache")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: service)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", service.download)
    return service


def test_dry_run_never_creates_a_hub_client(release, monkeypatch):
    import huggingface_hub

    def forbidden(*args, **kwargs):
        raise AssertionError("network client constructed during dry-run")

    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    plan = publish(release, "unit-test/release", public=True, dry_run=True)
    assert plan["decision_rows"] == 1000 and plan["visibility"] == "public"
    assert plan["dry_run"] is True
    assert not (release / "_state/hub_binding.json").exists()


def test_publication_provenance_payload_verification_marker_and_resume(release, hub):
    result = publish(release, "unit-test/release", public=True)
    mutations = [x for x in hub.calls if x[0] in ("create_commit", "upload_folder", "upload_file")]
    assert [x[0] for x in mutations] == ["create_commit", "upload_folder", "upload_file"]
    assert {op.path_in_repo for op in mutations[0][1]["operations"]} == {
        "run_config.json",
        "_PUBLICATION.json",
    }
    assert mutations[-1][1]["path_in_repo"] == "_SUCCESS.json"
    assert "_SUCCESS.json" not in mutations[1][1]["allow_patterns"]
    assert result["commit_sha"] == hub.head
    assert result["url"].endswith("/tree/" + hub.head)
    assert hub.trees[hub.head]["_SUCCESS.json"] == (release / "_SUCCESS.json").read_bytes()
    assert publish(release, "unit-test/release", public=True)["commit_sha"] == hub.head


@pytest.mark.parametrize("failure,match", [("skip", "missing"), ("corrupt", "checksum")])
def test_payload_not_marked_complete_if_ignored_or_corrupted(release, hub, failure, match):
    setattr(hub, failure, "schema.json")
    with pytest.raises(ValueError, match=match):
        publish(release, "unit-test/release", public=True)
    assert "_SUCCESS.json" not in hub.trees[hub.head]


def test_partial_upload_resumes_with_same_intent(release, hub):
    hub.fail_payload = True
    with pytest.raises(OSError, match="interrupted"):
        publish(release, "unit-test/release", public=True)
    assert "_PUBLICATION.json" in hub.trees[hub.head] and "_SUCCESS.json" not in hub.trees[hub.head]
    hub.fail_payload = False
    publish(release, "unit-test/release", public=True)
    assert "_SUCCESS.json" in hub.trees[hub.head]


@pytest.mark.parametrize("failure", ["fail_payload", "corrupt", "race_before_marker"])
def test_republication_invalidates_previous_marker_before_payload_changes(release, hub, failure):
    previous = publish(release, "unit-test/release", public=True)["commit_sha"]
    setattr(hub, failure, "schema.json" if failure == "corrupt" else True)
    with pytest.raises((OSError, ValueError)):
        publish(release, "unit-test/release", public=True)
    assert "_SUCCESS.json" not in hub.trees[hub.head]
    assert "_SUCCESS.json" in hub.trees[previous]
    setattr(hub, failure, None if failure == "corrupt" else False)
    result = publish(release, "unit-test/release", public=True)
    assert "_SUCCESS.json" in hub.trees[result["commit_sha"]]


def test_upload_plan_patterns_are_independent_of_publisher_state(release):
    plan = publish(release, "unit-test/release", dry_run=True)
    patterns = list(plan["allow_patterns"])
    plan["allow_patterns"].clear()
    assert publish(release, "unit-test/release", dry_run=True)["allow_patterns"] == patterns


@pytest.mark.parametrize("relative", ["_state", "_state/hub_binding.json", "_PUBLICATION.json"])
def test_publication_rejects_symlinked_intent_paths_before_network(release, monkeypatch, relative):
    import huggingface_hub

    external = release.parent / "external"
    linked = release / relative
    if relative == "_state":
        linked.rename(external)
    else:
        external.write_bytes(b"preserve this file")
    linked.symlink_to(external, target_is_directory=relative == "_state")

    def forbidden():
        raise AssertionError("network client must not be created for unsafe local paths")

    monkeypatch.setattr(huggingface_hub, "HfApi", forbidden)
    with pytest.raises(ValueError, match="symlink"):
        publish(release, "unit-test/release")
    if relative == "_state":
        assert not (external / "hub_binding.json").exists()
    else:
        assert external.read_bytes() == b"preserve this file"


def test_concurrent_branch_change_cannot_receive_success_marker(release, hub):
    hub.race_before_marker = True
    with pytest.raises(ValueError, match="parent commit conflict"):
        publish(release, "unit-test/release", public=True)
    assert "_SUCCESS.json" not in hub.trees[hub.head]


def test_additional_remote_data_rejected_after_upload(release, hub):
    hub.extra_after_payload = True
    with pytest.raises(ValueError, match="additional data"):
        publish(release, "unit-test/release", public=True)
    assert "_SUCCESS.json" not in hub.trees[hub.head]


def test_visibility_and_target_binding_not_silently_changed(release, hub):
    hub.private = True
    with pytest.raises(ValueError, match="visibility differs"):
        publish(release, "unit-test/release", public=True)
    hub.private = False
    publish(release, "unit-test/release", public=True)
    with pytest.raises(ValueError, match="different Hub target"):
        publish(release, "unit-test/another", public=True)


def test_mismatched_remote_intent_rejected(release, hub):
    publish(release, "unit-test/release", public=True)
    hub.commit(hub.head, {"_PUBLICATION.json": b"{}"})
    with pytest.raises(ValueError, match="different provenance or content"):
        publish(release, "unit-test/release", public=True)


def test_stale_validation_report_is_never_trusted(release, hub, monkeypatch):
    import thermoshift.validation as module

    atomic_json(
        release / "validation.json",
        {
            "status": "passed",
            "scope": "full",
            "manifest_sha256": read_json(release / "_SUCCESS.json")["manifest_sha256"],
        },
    )

    def reject(*args, **kwargs):
        raise ValueError("new scientific validation detects problem")

    monkeypatch.setattr(module, "_validate_locked", reject)
    with pytest.raises(ValueError, match="new scientific validation"):
        publish(release, "unit-test/release", public=True)
    assert hub.calls == []


def test_first_publication_creates_new_repository_with_requested_visibility(
    release, hub, monkeypatch
):
    from huggingface_hub.errors import RepositoryNotFoundError

    ready = False
    original_info = hub.repo_info

    def info(**kw):
        if not ready:
            raise RepositoryNotFoundError(
                "mock missing repository",
                response=httpx.Response(
                    404,
                    request=httpx.Request(
                        "GET", "https://huggingface.co/api/datasets/unit-test/release"
                    ),
                ),
            )
        return original_info(**kw)

    def create(**kw):
        nonlocal ready
        hub.check("create_repo", kw)
        assert kw["private"] is False and kw["exist_ok"] is False
        ready = True

    monkeypatch.setattr(hub, "repo_info", info)
    monkeypatch.setattr(hub, "create_repo", create, raising=False)
    publish(release, "unit-test/new-release", public=True)
    assert ready and "_SUCCESS.json" in hub.trees[hub.head]


def test_missing_branch_is_reported_before_upload(release, hub, monkeypatch):
    from huggingface_hub.errors import RevisionNotFoundError

    def missing(**kw):
        raise RevisionNotFoundError(
            "mock missing branch",
            response=httpx.Response(
                404,
                request=httpx.Request(
                    "GET", "https://huggingface.co/api/datasets/unit-test/release"
                ),
            ),
        )

    monkeypatch.setattr(hub, "repo_info", missing)
    with pytest.raises(ValueError, match="create the branch"):
        publish(release, "unit-test/release", public=True, revision="missing")
    assert hub.head == "initial"


def test_missing_remote_head_is_rejected_before_mutation(release, hub, monkeypatch):
    monkeypatch.setattr(hub, "repo_info", lambda **kwargs: SimpleNamespace(private=False, sha=None))
    with pytest.raises(ValueError, match="head SHA"):
        publish(release, "unit-test/release", public=True)
    assert hub.head == "initial"
    assert hub.calls == []
