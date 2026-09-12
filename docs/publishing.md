# Publishing and consumption

[Project overview](index.md) · [Generation](generation.md) · [Python API](api.md)

## Prepare the release

Install the `hub` extra and authenticate in the environment used to run the
command or Python API:

```bash
python -m pip install -e ".[hub]"
hf auth login
```

The SDK also reads `HF_TOKEN` from the environment. Keep credentials in your local
authentication configuration or secret manager.

A release must be finalized before publishing. Review the local upload plan first:

```bash
python -m thermoshift publish output/demo \
  --repo-id YOUR_USERNAME/thermoshift --public --dry-run
```

Replace `YOUR_USERNAME` with your Hugging Face namespace. The plan includes record
counts, compressed bytes, file patterns, visibility, branch, and publication steps.
A dry run performs full local validation and returns before constructing a Hub client.

## Publish

```bash
python -m thermoshift publish output/demo \
  --repo-id YOUR_USERNAME/thermoshift --public
```

The default visibility is private. Existing repository visibility must match the
selected mode. A new repository starts on `main`; use an existing branch with
`--revision` when publishing to another revision.

An upload returns the plan fields plus `repo_id`, `revision`, `commit_sha`, and
the completed revision URL. A dry run returns only the local plan, including
record counts, compressed bytes, visibility, file patterns, and steps; it has no
`commit_sha` or URL and does not upload. Record the commit SHA from a completed
upload alongside experiment results. The Python signature and return contracts
are in the [Python API](api.md).

The generated dataset card defines two configurations, `logged` and `oracle`, and
the nonempty splits in each. The payload consists of ordinary Parquet files,
configuration, schemas, feature roles, manifest, validation report, model reference,
and license. Local generation locks and shard-state files stay local.

## Retries and release identity

Repeat the same command to retry an interrupted upload. The local output and
remote publication intent bind the exact manifest, metadata, repository, branch,
and visibility. Use a separate output directory and a fresh repository or suitable
branch for a different release.

The publisher records provenance, uploads the payload, and verifies remote file
sizes and Git/LFS checksums at the returned immutable commit. It writes the
completion marker after verification. Parent-commit guards detect intervening
branch changes. Keep other writers away from the branch during publication.
Republishing clears any previous completion marker in the guarded provenance
commit before uploading payload files. A failed retry therefore leaves the branch
unmarked; previously completed immutable commit URLs remain available.

| Error | Resolution |
| --- | --- |
| Dataset is busy | Finish active generation or close local reading iterators |
| Existing visibility differs | Use the matching visibility or a separate repository |
| Remote branch has different content | Use a fresh release repository or branch |
| Local output is bound elsewhere | Resume its original target or create another local release |
| Missing or mismatched remote payload | Check upload ignore rules and rerun the same publication |
| Parent-commit conflict | Inspect the branch change before retrying |
| Branch does not exist | Create the branch first, or use `main` for a new repository |

## Load a published release

Use the repository ID and commit SHA from the publication result:

```python
from datasets import load_dataset

records = load_dataset(
    "YOUR_USERNAME/thermoshift",
    name="logged",
    split="train",
    revision="COMMIT_SHA_FROM_PUBLISH_OUTPUT",
    streaming=True,
    columns=["row_id", "building_id", "step", "obs_temp_last_c", "action", "y_next_temp_c"],
)

for row in records.take(5):
    print(row)
```

`streaming=True` returns an iterable dataset, and Parquet column projection selects
the requested fields. The same interface accepts a local release directory. See
the [Datasets streaming guide](https://huggingface.co/docs/datasets/stream) for
iterable operations and filtering.

Load oracle labels with `name="oracle"` and join using `row_id`. Match identifiers
explicitly when the streams have been shuffled or partitioned separately.

## Shuffle or distribute training data

For row-based learning, use a bounded shuffle buffer before taking a subset:

```python
shuffled = records.shuffle(seed=42, buffer_size=10_000)
```

For sequential learning, retain complete ordered building trajectories. Keep the
provided building splits when measuring held-out performance.

A distributed training job can partition the iterable with its own rank settings:

```python
from datasets.distributed import split_dataset_by_node

local_records = split_dataset_by_node(records, rank=0, world_size=4)
```

Training ranks partition an existing dataset. Generation ranks assign work while
creating that dataset. Use the respective rank settings for each stage.

## Further reference

The publisher uses [HfApi](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
for repository, upload, and immutable-tree operations. Plan repository capacity
using the measured Parquet size and the destination account's
[storage policy](https://huggingface.co/docs/hub/storage-limits).
