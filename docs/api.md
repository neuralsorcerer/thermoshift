# Python API

[Project overview](index.md) · [Generation](generation.md) · [Architecture](architecture.md)

## Create and validate

```python
from thermoshift import Config, generate, initialize, validate

config = Config(rows=100_000, seed=42, episode_steps=168)
initialize("output/python-example", config)
generation = generate("output/python-example", batch_buildings=256)
report = validate("output/python-example")

print(generation["written"], report["rows"], report["status"])
```

| Function | Result | Behavior |
| --- | --- | --- |
| `initialize(root, config)` | `Config` | Create an empty output plan or reopen a matching one |
| `generate(root, workers=1, batch_buildings=512, rank=0, world_size=1, progress=None)` | Summary dictionary | Generate or resume assigned shards; finalize single-rank runs |
| `finalize(root)` | Manifest dictionary | Verify all shards and write completed release metadata |
| `validate(root, full=True, batch_size=65536, write_report=True, replay=False)` | Validation dictionary | Check a stable release and optionally save the result |
| `validated_manifest(root, verify_hash=True)` | `(Config, manifest)` | Verify finalized file metadata and checksums |
| `iter_pairs(root, split=None, batch_size=65536)` | Iterator of `(file, logged, oracle)` | Stream matching Arrow batches in numeric shard order within each split |
| `publish(root, repo_id, public=False, revision="main", dry_run=False)` | Publication dictionary | Validate, upload, verify, and return the completed revision |

Paths accept strings and `pathlib.Path`. Boolean options accept `True` or `False`.
Invalid arguments and failed record checks raise `ValueError`; file operations can
raise `OSError`. The CLI converts these errors into a message and exit code 2.

## Use multiple processes

Put process-spawning code behind a main guard in a Python file:

```python
from thermoshift import Config, generate, initialize


def main():
    initialize("output/parallel", Config(rows=2_000_000, shard_buildings=4096))
    generate("output/parallel", workers=4, batch_buildings=256)


if __name__ == "__main__":
    main()
```

A progress callback receives a JSON string for each completed shard:

```python
import json

from thermoshift import generate


def on_shard(message):
    event = json.loads(message)
    print(event["shard_id"], event["status"])


generate("output/python-example", progress=on_shard)
```

Python generation is quiet by default. Callback execution happens in the calling
process. Finalization is explicit when `world_size` is greater than one.

## Read paired Arrow batches

```python
from contextlib import closing

from thermoshift import iter_pairs

with closing(iter_pairs("sample_dataset", split="test", batch_size=1024)) as batches:
    for file_info, logged, oracle in batches:
        assert logged["row_id"].equals(oracle["row_id"])
        print(file_info["split"], logged.num_rows)
        break
```

The iterator holds an exclusive snapshot lease. Exhaust it or close it when done;
`contextlib.closing` releases it on early exit. Returned objects are PyArrow tables.
File integrity is verified before the first batch is yielded.

## Simulate a building range

```python
from thermoshift import Config, simulate

config = Config(rows=1680, episode_steps=168)
logged, oracle, split_codes = simulate(config, start_building=0, stop_building=10)
print(logged.num_rows, oracle.num_rows)
```

The range is half-open and must lie inside `config.buildings`. This function
returns in-memory tables for that range. Its memory use grows with range size and
episode length; the generation workflow supplies bounded ranges automatically.

## Select features

```python
from thermoshift.schema import feature_roles

roles = feature_roles()
policy_columns = roles["policy_features"]
transition_columns = roles["transition_features"]
```

Policies receive observed decision-time features. Transition predictors additionally
receive `action`. Factual outcomes and oracle columns serve as targets, weights, or
evaluation information according to the task. The returned feature lists can be
modified independently of the package defaults.

## Evaluate a policy or fit the baseline

```python
from thermoshift.evaluation import evaluate_policy

result = evaluate_policy("sample_dataset", split="test")
print(result["ips"], result["snips"], result["oracle_policy_value"])
```

The fixed policy requests full cooling when the observed setpoint gap exceeds
1.5 °C, half cooling above −0.3 °C, and zero cooling otherwise. Estimates concern
one-step reward on the logged state distribution. `independent_buildings` is the
cluster count; `row_weight_ess_diagnostic` summarizes importance-weight concentration.
SNIPS is undefined when there are no matched actions. Interval fields are `None`
when the calculation lacks the required overlap or building count.

```python
from thermoshift.baseline import train_baseline

scores = train_baseline("sample_dataset", max_train=100_000, max_eval=100_000)
print(scores["splits"]["test"])
```

The baseline requires the `analysis` extra. It uses a bounded training prefix,
fixed gradient-boosting settings, and the transition feature list, with sine/cosine
hour encoding. Returned results include model and persistence MAE/RMSE by split.

## Inspect publication without uploading

```python
from thermoshift import publish

plan = publish("sample_dataset", repo_id="YOUR_USERNAME/thermoshift", dry_run=True)
print(plan["decision_rows"], plan["parquet_bytes"], plan["visibility"])
```

A dry run performs local validation and returns the plan. A completed upload also
returns `commit_sha` and the revision URL. See [publishing](publishing.md) for the
repository and authentication workflow.
