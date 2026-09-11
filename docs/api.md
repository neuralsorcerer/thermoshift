# Python API

[Project overview](index.md) · [Generation](generation.md) · [Schema](schema.md) · [Publishing](publishing.md)

This page documents the supported Python API. The package-level imports in
`thermoshift` are the recommended entry points. Module-level APIs listed below
are supported when a workflow needs more control or analysis. Names beginning
with `_`, and helpers listed as internal at the end of this page, are not stable
application interfaces.

## Package exports

```python
import thermoshift

print(thermoshift.__version__)
```

The package exports `Config`, `simulate`, `initialize`, `generate`, `finalize`,
`validate`, `iter_pairs`, `validated_manifest`, and `publish`, in addition to
`__version__`.

## Configure a release

`Config` is an immutable dataclass. Its complete constructor is:

```python
Config(
    rows=1_000_000,
    seed=42,
    episode_steps=168,
    shard_buildings=16_384,
    exploration=0.15,
    hidden_confounding=False,
    comfort_weight=0.30,
    carbon_weight=0.05,
    compression="zstd",
    compression_level=3,
    schema_version="2.0",
)
```

| Field | Contract |
| --- | --- |
| `rows` | Exact positive signed-64-bit decision count. |
| `seed` | Integer in `[0, 2**32)`. |
| `episode_steps` | Integer in `[2, 8760]`; the configured trajectory length in hours. |
| `shard_buildings` | Integer in `[1, 1_000_000]`. |
| `exploration` | Finite real number in `(0, 1]`; uniform action-policy mixture. |
| `hidden_confounding` | Strict boolean; when true, latent temperature influences logging. |
| `comfort_weight`, `carbon_weight` | Finite real numbers in `[0, 1_000_000)`. |
| `compression` | Currently only `"zstd"`. |
| `compression_level` | Integer in `[1, 19]`. |
| `schema_version` | Currently only `"2.0"`. |

Booleans are not accepted as integers, and numeric strings are not coerced.
Invalid values raise `ValueError`. Derived properties are `buildings` (the
ceiling of `rows / episode_steps`), `shards` (the ceiling of
`buildings / shard_buildings`), and `fingerprint` (the SHA-256 fingerprint of
the serialized configuration). `bounds(shard_id)` returns the half-open
`(start_building, stop_building)` range for a shard and raises `ValueError` for
an invalid shard ID.

```python
from thermoshift import Config

config = Config(rows=10_000, seed=42)
print(config.buildings, config.shards, config.fingerprint)
print(config.bounds(0))
```

## Generate, finalize, and validate

```python
from thermoshift import Config, finalize, generate, initialize, validate

root = "output/python-example"
config = initialize(root, Config(rows=100_000, seed=42, episode_steps=168))
generation = generate(root, batch_buildings=256)
report = validate(root)
print(config.fingerprint, generation["written"], report["status"])
```

### `initialize(root, config) -> Config`

Creates an empty output directory and writes the immutable configuration,
provenance, schema, and feature-role documents. Calling it again is allowed only
when the existing plan and runtime match. `config` must be a `Config` instance;
wrong types raise `TypeError`, while a nonempty or mismatched directory raises
`ValueError`.

### `generate(root, workers=1, batch_buildings=512, rank=0, world_size=1, progress=None) -> dict`

Generates or resumes the shards assigned to `rank` and returns:

```python
{"written": 3, "resumed": 1, "seconds": 12.34}
```

`workers`, `batch_buildings`, and `world_size` must be positive integers;
`rank` must be in `[0, world_size)`. `progress` is either `None` or callable.
The callback runs in the calling process and receives one JSON string per
completed shard, with `shard_id`, `status` (`"written"` or `"resumed"`), and
`seconds` fields. With `world_size=1`, generation automatically calls
`finalize`. With multiple ranks, call `finalize` after every rank completes.

Put process-spawning code behind a main guard:

```python
from thermoshift import Config, generate, initialize


def main():
    root = "output/parallel"
    initialize(root, Config(rows=2_000_000, shard_buildings=4096))
    generate(root, workers=4, batch_buildings=256)


if __name__ == "__main__":
    main()
```

### `finalize(root) -> dict`

Requires every shard to be committed and verifies shard files, row accounting,
and checksums before writing `manifest.json`, the generated cards, and
`_SUCCESS.json`. It returns the complete manifest with `complete`, `fingerprint`,
`rows`, `buildings`, `split_rows`, `bytes`, and sorted `files` entries. An
incomplete or inconsistent release raises `ValueError`.

### `validate(root, full=True, batch_size=65536, write_report=True, replay=False) -> dict`

Validates a finalized release. Metadata-only validation uses `full=False`; full
validation also checks record semantics, trajectory continuity, probabilities,
physical equations, reward accounting, and oracle labels. `replay=True` requires
`full=True` and regenerates every building using the initialized source and
runtime. `batch_size` must be a positive integer, and `full`, `write_report`,
and `replay` must be strict booleans.

A successful report contains `status="passed"`, `scope`, `fingerprint`,
`manifest_sha256`, `rows`, `buildings`, `files`, `bytes`, `findings`,
`split_profiles`, `replay`, validator provenance, and `seconds`. Full reports
also contain the record-check list and replay reports contain the replay check.
With `write_report=True` (the default), `validation.json` is written with
`running`, then `passed`, `failed`, or `interrupted` status. `False` leaves that
file unchanged. Failed checks raise `ValueError` after writing failure details
when report writing is enabled.

## Read verified pairs

### `validated_manifest(root, verify_hash=True) -> (Config, dict)`

Verifies the finalized release metadata, manifest structure, file sizes, row
counts, and Arrow schemas. With `verify_hash=True`, it also verifies every
Parquet SHA-256 checksum; `False` skips those individual file hashes but still
performs the other structural checks. Malformed or inconsistent release
metadata raises `ValueError`; a missing path or file can raise an `OSError`
such as `FileNotFoundError`. `verify_hash` must be a strict boolean.

### `iter_pairs(root, split=None, batch_size=65536) -> iterator`

Yields `(file_info, logged, oracle)` in numeric shard order. `logged` and
`oracle` are PyArrow tables with matching rows and `row_id`s; `file_info` is the
manifest entry for the logged file. `split=None` reads all splits, while a split
name restricts the iterator to that split. The iterator verifies checksums before
the first batch and holds an exclusive lifecycle lease until exhausted or
closed. Close it when stopping early:

```python
from contextlib import closing
from thermoshift import iter_pairs

with closing(iter_pairs("output/demo", split="test", batch_size=1024)) as pairs:
    for file_info, logged, oracle in pairs:
        assert logged["row_id"].equals(oracle["row_id"])
        print(file_info["split"], logged.num_rows)
        break
```

Unknown splits and nonpositive batch sizes raise `ValueError`.

## Simulate in memory

### `simulate(config, start_building, stop_building) -> (pa.Table, pa.Table, numpy.ndarray)`

Simulates the half-open building range `[start_building, stop_building)` and
returns `(logged, oracle, split_codes)`. The first two values are PyArrow tables
with the schemas in [schema.md](schema.md); the third is a NumPy integer array
containing one split code per returned row (`0=train`, `1=validation`, `2=test`,
`3=test_heatwave`, `4=test_sensor`). Memory is bounded by the selected range,
and only the final configured trajectory can be truncated to achieve the exact
`rows` count. Invalid ranges raise `ValueError`.

`thermal_step(temp, outdoor, conductance, capacity, heat_kw, cooling_kw)` is the
vectorized one-hour exact constant-forcing RC step used by the simulator. It
returns the next temperature as a NumPy-compatible scalar or array. Temperatures
are in degrees Celsius, conductance in kW/degree Celsius, capacity in
kWh/degree Celsius, and heat/cooling in kW.

```python
from thermoshift import Config, simulate
from thermoshift.simulator import thermal_step

config = Config(rows=1680, episode_steps=168)
logged, oracle, split_codes = simulate(config, 0, 10)
assert logged.num_rows == oracle.num_rows == len(split_codes)
next_temp = thermal_step(24.0, 30.0, 0.5, 10.0, 1.0, 2.0)
```

## Schemas and feature roles

`thermoshift.schema` exposes `LOGGED_SCHEMA`, `ORACLE_SCHEMA`, `SCHEMAS`, and
`FEATURES` as PyArrow schema/list constants. The callable helpers are:

```python
from thermoshift.schema import feature_roles, schema_document

document = schema_document()
roles = feature_roles()
print(roles["policy_features"])
```

`schema_document()` returns the JSON-serializable description of both schemas,
including names, Arrow types, nullability, roles, units, and descriptions.
`feature_roles()` returns independent lists under `policy_features`,
`transition_features` (policy features plus `action`), and
`forbidden_policy_inputs`, plus the `oracle_access` usage string. The generated
`schema.json` and `feature_roles.json` are serialized copies of these results.
See [schema.md](schema.md) for every column.

## Evaluation and baseline

### `evaluate_policy(root, split="test") -> dict`

```python
from thermoshift.evaluation import evaluate_policy

result = evaluate_policy("output/demo", split="test")
print(result["ips"], result["snips"], result["oracle_policy_value"])
```

This evaluates the fixed threshold policy on the logged-state distribution using
the supplied propensities. The result includes `split`, `rows`,
`independent_buildings`, `estimand`, `overlap_status`, ratio objects for `ips`,
`snips`, `oracle_policy_value`, `logged_policy_value`, and
`oracle_one_step_regret`, `row_weight_ess_diagnostic`, and `notes`. Each ratio
object has `value`, `se_building_cluster`, and `ci95_normal_approx`; fields are
`None` when overlap or cluster count is insufficient. It requires a finalized,
nonempty split and raises `ValueError` for invalid or empty data. Missing release
paths or files can raise an `OSError`.

### `train_baseline(root, max_train=100000, max_eval=100000) -> dict`

Requires the `analysis` extra. It fits the bounded HistGradientBoosting
temperature baseline using policy features plus `action`, and reports model and
persistence MAE/RMSE for each nonempty evaluation split. `max_train` must be at
least 10 and `max_eval` at least 1. The result has `training_rows`, `features`,
`oracle_used_for_training=False`, and a `splits` mapping. A release with fewer
than ten training rows raises `ValueError`.

```python
from thermoshift.baseline import train_baseline

scores = train_baseline("output/demo", max_train=100_000, max_eval=100_000)
print(scores["splits"].get("test"))
```

`ClusterMoments` is the streaming building-cluster accumulator used by policy
evaluation. It accepts ordered `(n, 6)` value batches with `add(ids, values)`,
requires `flush()` before `ratio(numerator, denominator)`, and supports ratio
pairs `(1,0)`, `(1,2)`, `(3,0)`, `(4,0)`, and `(5,0)`. It is available for
advanced analysis but its internal accumulator fields are not a stable result
format.

## Publication

### `publish(root, repo_id, public=False, revision="main", dry_run=False) -> dict`

Requires the `hub` extra. It performs a full local validation and publishes a
verified immutable dataset tree to Hugging Face. `public` and `dry_run` must be
strict booleans; `repo_id` must be `OWNER/DATASET`; `revision` must be a
nonempty string.

With `dry_run=True`, the result is a plan containing `repo_id`, `revision`,
`visibility`, `decision_rows`, `parquet_files`, `parquet_bytes`,
`allow_patterns`, `upload_method`, `dry_run`, and `steps`. No network upload or
`commit_sha` is produced. A completed upload returns the same plan fields plus
`commit_sha` and `url`. Publication failures include missing extras, invalid
repositories or revisions, visibility/provenance conflicts, and remote payload
or parent-commit mismatches.

## Notebook execution

`thermoshift.notebook.execute(path, working_directory) -> int` executes trusted
plain-Python code cells in order in the current process, captures stdout/stderr
and final-expression display output, saves the notebook atomically after each
cell, and returns the number of code cells executed. It changes into
`working_directory` during execution and restores the original directory. Syntax,
execution, and interruption exceptions are saved in the notebook and re-raised.

```python
from thermoshift.notebook import execute

cells_executed = execute("notebooks/quickstart.ipynb", ".")
```

Only execute notebooks from a trusted source: cells have normal in-process
Python access to the host environment.

## Command-line API

Both `thermoshift` and `python -m thermoshift` invoke the same CLI. Use
`thermoshift --version` for the installed version and `--help` for parser help.
Commands are `init`, `plan`, `generate`, `finalize`, `validate`, and `publish`.
Their options and operational guidance are in [generation.md](generation.md) and
[publishing.md](publishing.md). Successful commands print JSON to stdout;
progress is written to stderr. Handled `ValueError`/`OSError` failures return 2,
and interruption returns 130. Parser errors are handled by `argparse` and do not
produce a JSON result.

The callable `thermoshift.cli.main(argv=None) -> int` runs the CLI and returns
the status for handled operation failures and successful commands. Argument
parsing happens before that operation handling; invalid command-line syntax
raises `SystemExit` through `argparse` rather than returning an integer.
`thermoshift.cli.parser()` builds the parser for embedding or testing. The JSON
result shapes are:

| Command | Result |
| --- | --- |
| `init` | `rows`, `buildings`, `shards`, `fingerprint` |
| `plan` | `rows`, `buildings`, `shards`, `parquet_files_upper_bound`, `simulation_rows_per_batch_upper_bound` |
| `generate` | `written`, `resumed`, `seconds` |
| `finalize` | `rows`, `files`, `bytes` |
| `validate` | Validation report described above |
| `publish` | Publication plan/result described above |

## Internal implementation helpers

The following importable helpers support package internals and tests but are not
stable user APIs: `config.integer`, `config.boolean`; filesystem atomic-write,
hash, and JSON helpers; provenance fingerprint/loading helpers; random stream
helpers; shard path/state helpers; lifecycle locking; generated-card writing;
and `reading.ordered_files`. Prefer the package exports and documented module
APIs above. Private names such as `_validated_manifest`, `_iter_pairs`,
`_read_bounded`, and `_run_locked` are implementation details.
