<h1 align="center">
ThermoShift
</h1>
<h3 align="center">
Synthetic building-cooling trajectories with paired counterfactual outcomes for machine learning.
</h3>

---

<div align="center">

[![Current Release](https://img.shields.io/github/release/neuralsorcerer/thermoshift.svg)](https://github.com/neuralsorcerer/thermoshift/releases)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-fcbc2c.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Test Linux](https://github.com/neuralsorcerer/thermoshift/actions/workflows/ubuntu.yml/badge.svg?branch=main)](https://github.com/neuralsorcerer/thermoshift/actions/workflows/ubuntu.yml?query=branch%3Amain)
[![Test Windows](https://github.com/neuralsorcerer/thermoshift/actions/workflows/windows.yml/badge.svg?branch=main)](https://github.com/neuralsorcerer/thermoshift/actions/workflows/windows.yml?query=branch%3Amain)
[![Test macOS](https://github.com/neuralsorcerer/thermoshift/actions/workflows/macos.yml/badge.svg?branch=main)](https://github.com/neuralsorcerer/thermoshift/actions/workflows/macos.yml?query=branch%3Amain)
[![Lint](https://github.com/neuralsorcerer/thermoshift/actions/workflows/lint.yml/badge.svg?branch=main)](https://github.com/neuralsorcerer/thermoshift/actions/workflows/lint.yml?query=branch%3Amain)
[![CodeQL](https://github.com/neuralsorcerer/thermoshift/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/neuralsorcerer/thermoshift/actions/workflows/codeql.yml?query=branch%3Amain)
[![Documentation](https://github.com/neuralsorcerer/thermoshift/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/neuralsorcerer/thermoshift/actions/workflows/docs.yml?query=branch%3Amain)
[![License](https://img.shields.io/badge/License-MIT-3c60b1.svg?logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>


ThermoShift generates hourly cooling decisions for temperature prediction,
action-effect estimation, one-step policy evaluation, and sensor-state modeling.
Each decision includes observed context, a sampled action, its logging probability,
and the resulting temperature, electricity use, cost, emissions, and comfort
penalty. A paired oracle table supplies latent state and potential outcomes for
all three cooling actions.

The Python package and CLI write sharded Parquet releases with deterministic
building splits, resumable generation, checksums, validation reports. Run it locally or as a batch job; multi-machine generation
uses a shared filesystem.


## Capabilities and scope

| Capability | Implementation |
| --- | --- |
| Paired supervision | One `logged` row and one `oracle` row per decision, joined by `row_id` |
| Ordered trajectories | One trajectory per synthetic building; exact requested decision count |
| Distribution shifts | Held-out buildings plus heatwave and sensor-degradation test splits |
| Reproducible simulation | Indexed random streams; numeric records independent of worker and batch scheduling in the same source/runtime |
| Bounded generation | Vectorized building batches, process workers, and bounded task scheduling |
| Recovery | Per-shard state, atomic file replacement, SHA-256 verification, regeneration of incomplete or corrupt shards |
| Release validation | File inventory, schemas, record consistency, physical equations, and optional exact replay |
| Consumption and sharing | Paired PyArrow batches, Hugging Face streaming, verified publication to a pinned commit |
| Reference experiments | Temperature regression baseline and propensity-based one-step policy evaluation |

The simulator is an idealized, single-zone research model. Its validation checks
internal consistency with the specified model; it does not establish accuracy for
real buildings or qualify a controller for deployment. Counterfactual labels cover
one hour from a logged state. Evaluating a replacement policy over an entire
trajectory requires simulating the states that policy would produce.

## Installation

Browse the [documentation guides](docs/index.md), or see
[development and documentation](docs/development.md) to build the Sphinx site locally.

Use Python 3.11 or later. Core dependencies are NumPy, PyArrow,
filelock, portalocker, and PyYAML; version ranges are defined in
[pyproject.toml](pyproject.toml).

From a source checkout:

```bash
git clone https://github.com/neuralsorcerer/thermoshift.git
cd thermoshift
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[hub,analysis]"
python -m thermoshift --version
python -m thermoshift --help
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Run the examples
below from the repository root with that environment active. Both `thermoshift`
and `python -m thermoshift` invoke the same CLI.

For a generator-only installation, use `python -m pip install .`. For a deployed
batch environment, install a built wheel and retain the exact source and dependency
versions used to initialize each run. Editable installs are convenient for local
development, but package edits change the generation fingerprint.

| Extra | Install from the repository | Enables |
| --- | --- | --- |
| `hub` | `python -m pip install ".[hub]"` | Hugging Face publishing and Datasets loading |
| `analysis` | `python -m pip install ".[analysis]"` | scikit-learn temperature baseline and pandas analysis |
| `notebook` | `python -m pip install ".[notebook]"` | Notebook format, kernel, and display dependencies |
| `dev` | `python -m pip install -e ".[dev]"` | Tests, linting, formatting, scientific checks, and distribution builds |

Combine extras as needed. [constraints.txt](constraints.txt) records development
dependency versions; apply it with `-c constraints.txt` when reproducing that
setup. It is a constraints file, not a complete lock of every transitive dependency.

## Quickstart

Create and validate a small release before scaling up:

```bash
python -m thermoshift init output/demo --rows 10000 --seed 42
python -m thermoshift plan output/demo
python -m thermoshift generate output/demo
python -m thermoshift validate output/demo
```

This creates exactly **10,000 decisions across 60 buildings**, using the default
168-hour episode length. The final building has 88 stored steps. Each decision
appears once in each table, so the release contains 10,000 `logged` rows and 10,000
`oracle` rows. `generate` automatically finalizes a single-rank run; a successful
full validation returns JSON containing `"status": "passed"`, `"scope": "full"`,
and `"rows": 10000`.

The equivalent Python workflow is:

```python
from thermoshift import Config, generate, initialize, validate

initialize("output/python-demo", Config(rows=10_000, seed=42))
generation = generate("output/python-demo", batch_buildings=256)
report = validate("output/python-demo")
assert report["status"] == "passed"
print(generation["written"], report["rows"])
```

Repeat `generate` to resume a run, then validate again. Use a new directory when
changing the configuration, source, or recorded runtime. See
[production operations](#production-operations) for recovery and release handling.

## Dataset contract

### Tables and identifiers

| Configuration | Contents | Intended use |
| --- | --- | --- |
| `logged` | Decision-time observations, action, action probabilities, factual targets, episode boundary | Observed-feature training and propensity-based estimation |
| `oracle` | True temperature, physical parameters, sensor labels, six potential outcomes per action, best one-step action | Evaluation or explicitly declared privileged supervision |

The common identifiers are `row_id`, `building_id`, and `step`:

```text
row_id = building_id * episode_steps + step
```

`row_id` is a signed 64-bit identifier unique within a release. It is not a global
identifier across experiments. Records are ordered by building and step within
partitions; the paired reader visits shards in numeric order within each split.
`episode_end` marks the final stored step, including a shortened final trajectory.

| Field group | Examples | Meaning |
| --- | --- | --- |
| Observations | `outdoor_temp_c`, `occupancy_count`, `price_per_kwh`, `grid_available` | Context available at the decision time |
| Sensor state | `obs_temp_c`, `obs_temp_last_c`, `sensor_age_steps` | Current nullable reading, latest available reading, and reading age |
| Action | `action` | `0`: off; `1`: 50% rated electrical input; `2`: 100% rated electrical input |
| Logging probabilities | `p_action_0`, `p_action_1`, `p_action_2`, `propensity` | Actual sampling probabilities; `propensity` is the selected action's probability |
| Factual outcomes | `y_next_temp_c`, `y_energy_kwh`, `y_cost`, `y_carbon_kg`, `y_discomfort_c`, `y_reward` | Observed transition under the sampled action |
| Oracle outcomes | `cf_next_temp_c_0`, `cf_reward_0`, corresponding fields for actions `1` and `2` | Potential outcomes sharing the current state and hourly disturbances |

Physical calculations use float64; measured features and outcomes are stored as
float32, while action probabilities remain float64. Sensor dropout is an Arrow
null in `obs_temp_c`. `obs_temp_last_c` retains the latest reading and falls back
to the setpoint if the episode begins with dropout.

[docs/schema.md](docs/schema.md) lists every column, type, unit, role, and nullability.
Generated `schema.json` and `feature_roles.json` provide machine-readable contracts.

### Splits

A hash of building ID and seed assigns each building to exactly one split. The
percentages below describe hash-bucket allocation; realized building and row
counts vary and are recorded in `manifest.json`. Very small releases may have
empty splits, which are omitted from the generated loading configuration.

| Split | Allocation | Conditions |
| --- | ---: | --- |
| `train` | 70% | Standard generating conditions |
| `validation` | 10% | Standard conditions, disjoint buildings |
| `test` | 10% | Standard conditions, disjoint buildings |
| `test_heatwave` | 5% | Outdoor offset +8 °C, COP multiplied by 0.85, increased outage-entry probability |
| `test_sensor` | 5% | Stronger sensor drift, more frequent and persistent dropout |

Preserve these building splits for held-out evaluation. Preserve complete ordered
trajectories for sequence models; random row splitting leaks information between
hours from the same building.

### Release files

| Path | Purpose |
| --- | --- |
| `data/logged/<split>/<bucket>/part-*.parquet` | Observed decisions and factual outcomes |
| `data/oracle/<split>/<bucket>/part-*.parquet` | Matching latent state and counterfactual outcomes |
| `run_config.json` | Immutable configuration, config/source fingerprints, recorded runtime |
| `schema.json`, `feature_roles.json` | Types, units, roles, and feature lists |
| `manifest.json` | Exact file inventory, row counts, compressed sizes, SHA-256 hashes |
| `README.md`, `DATASHEET.md`, `LICENSE` | Generated dataset card, model reference, and license |
| `_SUCCESS.json` | Completion marker binding the manifest and release metadata |
| `validation.json` | Latest saved validation status, scope, and split profiles |
| `_state/shard-*.json` | Local committed-shard records used for recovery |
| `_PUBLICATION.json`, `_state/hub_binding.json` | Publication intent and local target binding, created when publishing |

Buckets group 1,000 shard IDs. Each shard writes one file per nonempty split per
configuration, with at most ten Parquet files per shard. Files use Zstandard
compression, level 3 by default.

## Load and use the data

### Stream with Hugging Face Datasets

Requires the `hub` extra:

```python
from datasets import load_dataset

records = load_dataset(
    "output/demo",
    name="logged",
    split="train",
    streaming=True,
    columns=["row_id", "building_id", "step", "obs_temp_last_c", "action", "y_next_temp_c"],
)

for record in records.take(3):
    print(record)
```

For a published release, replace the directory with `OWNER/DATASET` and pass
`revision="COMMIT_SHA_FROM_PUBLISH_OUTPUT"`. Load `name="oracle"` to access oracle
labels. Join by `row_id` when streams have been shuffled or partitioned separately.
External loaders read the Parquet data; run ThermoShift validation separately when
you need its release-integrity and scientific checks.

### Read verified pairs with PyArrow

The core package can stream both tables without the Hub extra:

```python
from contextlib import closing

from thermoshift import iter_pairs

with closing(iter_pairs("output/demo", split="test", batch_size=1024)) as batches:
    for file_info, logged, oracle in batches:
        assert logged["row_id"].equals(oracle["row_id"])
        print(file_info["split"], logged.num_rows)
        break
```

`iter_pairs` verifies release metadata and file checksums before yielding its first
batch. It holds an exclusive dataset lease until exhausted or closed; the context
manager releases it even when stopping early. It verifies file integrity but does
not run the full record-level validation scan.

### Select features and run reference experiments

Use the declared feature lists to keep targets and privileged state out of model
inputs:

```python
from thermoshift.schema import feature_roles

roles = feature_roles()
policy_columns = roles["policy_features"]
transition_columns = roles["transition_features"]  # Also includes the action.
```

```bash
python examples/train_baseline.py --data output/demo
python examples/evaluate_policy.py --data output/demo --split test
python examples/stream_hf.py output/demo --config logged --split train --limit 3
```

The temperature baseline requires `analysis`. It fits a gradient-boosted regressor
on a bounded training prefix and reports MAE/RMSE against persistence on each
nonempty evaluation split. Use `--max-train`, `--max-eval`, and `--report` to control
its limits and save results.

The policy example uses the core dependencies. It evaluates a fixed observed-gap
threshold policy with inverse propensity scoring (IPS), self-normalized IPS
(SNIPS), oracle policy value, and one-step regret. Uncertainty is clustered by
building. It reports missing intervals when overlap or building counts are
insufficient; row-weight effective sample size is an overlap diagnostic. Its
`--report` option saves the JSON result.

### Bundled sample and notebook

[sample_dataset/](sample_dataset/) contains 50,003 decisions from 298 buildings
and can be loaded by substituting `sample_dataset` in the loading examples. It is
included in the repository, but excluded from the wheel and source distribution.

The sample records a different generator fingerprint and a Python 3.12.14 runtime.
During verification with the current source on Python 3.13.9, its metadata and
checksum checks passed, but full validation failed with
`observable logging policy is not exactly reproducible`. Use a freshly generated,
fully validated release for experiments requiring the current validation contract;
the sample's saved report is not evidence of passing a new validation run.

The walkthrough explores the bundled sample's tables, splits, feature roles, and
paired outcomes:

```bash
python -m pip install -e ".[hub,analysis,notebook]"
python scripts/run_notebook.py
```

Open [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) to inspect it. The runner
executes trusted plain-Python cells in process and saves outputs into the notebook;
it is not a sandbox or a general Jupyter server.

## Simulation and scientific interpretation

Each building follows an hourly single-zone resistance–capacitance thermal model.
Let `T` and `T_out` be indoor and outdoor temperature, `G` conductance in kW/°C,
`C` heat capacity in kWh/°C, `Q` total internal/solar/residual heat in kW, `P` rated
electrical input in kW, `COP` cooling efficiency, and `g` grid availability. The
action selects an input fraction `a` in `{0, 0.5, 1}`. For constant hourly inputs:

```text
alpha = 1 - exp(-G * 1 hour / C)
T_next(a) = T + alpha * (T_out - T + (Q - COP * P * a * g) / G)
energy(a) = g * (background_load + P * a) * 1 hour
cost(a) = price * energy(a)
carbon(a) = carbon_intensity * energy(a)
discomfort(a) = max(abs(T_next(a) - setpoint) - 1, 0)
reward(a) = -(cost(a) + carbon_weight * carbon(a)
             + comfort_weight * occupancy_weight * discomfort(a)^2)
```

The occupancy weight is `1` when occupied and `0.1` otherwise. A grid outage sets
served electricity and cooling to zero. The sampled action's temperature advances
the trajectory; all counterfactual branches share that hour's initial state and
disturbances. `oracle_action` maximizes serialized one-step reward, breaking ties
with the smallest action index.

The logging policy mixes a temperature/price-dependent softmax with uniform
exploration. Each action has probability at least `exploration / 3` (0.05 at the
default setting). With `--hidden-confounding`, latent temperature also affects
assignment; use the supplied true propensities, since probabilities re-estimated
from observed features alone generally do not reproduce that assignment process.

These semantics matter when interpreting results:

- Policies use decision-time `policy_features`; outcome and oracle fields are
  targets or evaluation information unless privileged training is explicitly intended.
- Policy estimates describe **one-step reward on the logging-policy state
  distribution**, not the long-run value of a replacement controller.
- `episode_end` denotes stored trajectory truncation. Define termination and
  bootstrapping behavior explicitly when constructing a sequential-control task.
- Synthetic weather, occupancy, tariffs, emissions, and comfort penalties are
  illustrative assumptions. Internal equation checks do not measure real-world fidelity.

See the [model and parameter reference](src/thermoshift/resources/DATASHEET.md)
for the complete distributions, sensing process, logging equation, and units.

## CLI and configuration

Every subcommand accepts `--help` for its exact interface:

| Command | Purpose | Key options |
| --- | --- | --- |
| `init OUTPUT` | Create an immutable plan in an empty directory, or reopen a matching plan | Generation parameters below |
| `plan OUTPUT` | Report decisions, buildings, shards, maximum file count, and batch row bound | `--batch-buildings` |
| `generate OUTPUT` | Generate or resume assigned shards; finalize single-rank runs | `--workers`, `--batch-buildings`, `--rank`, `--world-size`, `--quiet` |
| `finalize OUTPUT` | Verify all shards and commit release metadata after multi-rank generation | — |
| `validate OUTPUT` | Check a finalized release and save a report | `--batch-size`, `--metadata-only`, `--replay`, `--no-report` |
| `publish OUTPUT` | Validate, upload, verify, and mark a Hub release complete | `--repo-id`, `--public`, `--revision`, `--dry-run` |

### Parameters fixed at initialization

| Option | Default | Constraint / meaning |
| --- | ---: | --- |
| `--rows` | `1000000` | Exact decisions; positive signed 64-bit integer |
| `--seed` | `42` | Integer in `[0, 2^32)` |
| `--episode-steps` | `168` | Hours per building, from 2 through 8,760 |
| `--shard-buildings` | `16384` | Buildings per shard, from 1 through 1,000,000 |
| `--exploration` | `0.15` | Uniform policy-mixture weight in `(0, 1]` |
| `--hidden-confounding` | Off | Include latent temperature in action assignment |
| `--comfort-weight` | `0.30` | Squared endpoint discomfort weight, in `[0, 1000000)` |
| `--carbon-weight` | `0.05` | Operational emissions weight, in `[0, 1000000)` |

The Python `Config` additionally exposes `compression="zstd"`,
`compression_level=3` (supported levels 1–19), and `schema_version="2.0"`.
The CLI does not expose compression flags. Do not edit `run_config.json` in place.

### Scheduling and automation

Generation defaults to `--workers 1 --batch-buildings 512 --rank 0 --world-size 1`.
Worker and batch counts must be positive; ranks satisfy `0 <= rank < world_size`.
Validation defaults to batches of 65,536 rows.

Successful commands write one final JSON result to stdout. Generation writes
per-shard JSON progress to stderr; `--quiet` suppresses that progress. For example,
after initializing `output/demo`:

```bash
python -m thermoshift generate output/demo \
  > output/demo-generation.json 2> output/demo-generation.log
```

The CLI returns `0` on success, `2` for argument errors and handled
`ValueError`/`OSError` failures, and `130` for a keyboard interruption. Automation
should treat every nonzero exit as failure; unexpected exceptions can have other
nonzero statuses. See [docs/api.md](docs/api.md) for return values, callbacks,
in-memory simulation, and Python multiprocessing examples.

## Production operations

### Size a run before scaling

```bash
python benchmarks/generation.py \
  --output output/pilot \
  --rows 2000000 \
  --workers 2 \
  --batch-buildings 512 \
  --shard-buildings 4096 \
  --report output/pilot-benchmark.json
```

The benchmark requires a fresh directory and measures generation, full validation,
compressed bytes per decision across both tables, and split profiles. Its memory
metric is parent-process peak RSS; measure total job memory separately when using
multiple workers. See [benchmarks/README.md](benchmarks/README.md).

Batch size controls simulation memory. Each worker simulates up to
`min(batch_buildings, shard_buildings) * episode_steps` transitions at once, with
additional Arrow, filtering, and compression allocations. Shard size controls file
count, parallel work, and how much must be rebuilt after a failure. Workers are
capped by assigned shard count: increasing workers cannot parallelize a one-shard
run. File inventories and finalization metadata grow with shard count.

For example, a billion-decision plan with the default episode and shard sizes has
5,952,381 buildings, 364 shards, and at most 3,640 Parquet files:

```bash
python -m thermoshift init output/large --rows 1000000000 --seed 42
python -m thermoshift plan output/large --batch-buildings 256
python -m thermoshift generate output/large --workers 8 --batch-buildings 256
python -m thermoshift validate output/large
```

Choose hardware and disk capacity from measured pilot results. `plan` reports
counts and batch bounds, not a throughput, disk-size, or RAM guarantee. Account
for temporary files and checksum/full-validation I/O as well as the final payload.

### Reproducibility and recovery

Initialization records a configuration fingerprint, a digest of package Python
files and packaged Markdown resources, and exact Python, ThermoShift, NumPy, and
PyArrow versions. Generation, finalization, and replay require that recorded source
and runtime. Preserve the built artifact and environment for the life of the run.

Indexed random draws make numeric records stable across worker counts, batch sizes,
and shard scheduling with the same scientific settings and runtime. Changing batch
boundaries can change Parquet row groups and compressed bytes, so numeric equality
does not imply identical file hashes. Increasing the requested size in a **new**
run preserves the existing decision prefix when seed, episode length, and model
settings match, except for the previously truncated `episode_end` marker.

To recover an interrupted local release:

```bash
python -m thermoshift generate output/large --workers 8 --batch-buildings 256
python -m thermoshift validate output/large
```

Completed shards are checksum-verified; incomplete or corrupt shards are rebuilt.
Parquet files are closed, flushed, atomically renamed, and hashed before shard state
is committed. Starting generation clears the previous completion marker and saved
validation report; finish generation and revalidate before consuming the release.
Keep `_state/` with the local output to retain its committed-shard recovery records.

### Multiple machines or job ranks

Use a shared filesystem supporting advisory locks and atomic same-directory
renames, with matching package source and recorded runtime on every rank.
Initialize once; shard `i` is assigned to rank `i % world_size`:

```bash
python -m thermoshift init /shared/thermoshift --rows 1000000000

# Run these as separate jobs against the same initialized directory.
python -m thermoshift generate /shared/thermoshift --world-size 2 --rank 0 --workers 8
python -m thermoshift generate /shared/thermoshift --world-size 2 --rank 1 --workers 8

# Only after every rank has finished successfully:
python -m thermoshift finalize /shared/thermoshift
python -m thermoshift validate /shared/thermoshift
```

Generation ranks hold shared lifecycle leases and per-shard locks. Finalization,
validation, publication, and paired readers require exclusive leases. Conflicting
operations fail immediately with a busy error. Close readers before starting
another exclusive operation, and ensure all writers follow the locking protocol.
Generation takes filesystem paths; generate locally or on the shared filesystem
before uploading to remote storage.

### Validation and release handoff

```bash
# Full file-integrity and scientific record checks; writes validation.json.
python -m thermoshift validate output/demo

# Hashes, inventory, schemas, and counts; no scientific record scan.
python -m thermoshift validate output/demo --metadata-only

# Full checks plus exact regeneration of every building.
python -m thermoshift validate output/demo --replay

# Run full checks without replacing the saved report.
python -m thermoshift validate output/demo --no-report
```

`--metadata-only` still reads file bytes to verify hashes. `--replay` requires full
validation and cannot be combined with `--metadata-only`. Exact replay adds work
proportional to regenerating the release, one building at a time.

Full validation checks IDs, split membership, trajectory and sensor continuity,
logging probabilities, factual/oracle alignment, physical domains, thermal and
electrical equations, reward accounting, cooling monotonicity, and oracle labels.
Reports use `running`, `passed`, `failed`, or `interrupted` status and record the
check scope. A killed process can leave `running`; rerun validation to refresh it.
Each report-writing run replaces the previous report, including metadata-only runs.

For a handoff, retain `_SUCCESS.json`, its bound manifest and metadata, the full
payload, and a passing **full-scope** validation report. The completion marker
records finalization; it does not itself certify a full scientific validation.
Treat finalized release files as immutable and validate transferred copies before
use. Publication performs a fresh full local validation regardless of a saved report.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| Missing command options or imports after updating the checkout | Confirm the active environment with `python -c "import thermoshift; print(thermoshift.__file__)"`; reinstall this checkout with `python -m pip install -e ".[hub,analysis]"`. An older installed package can otherwise take precedence. |
| Output contains a different configuration / new output must be empty | Initialize a new empty directory with the intended parameters. |
| Generator code/runtime differs from initialization | Restore the recorded source and exact runtime, or start a new release directory. |
| `dataset is busy` | Finish active operations and exhaust or close paired iterators before retrying. |
| Shard is incomplete during finalization | Complete all ranks, then run `finalize` and full validation. |
| Parquet checksum fails | Resume generation in the original environment to rebuild the affected shard, then validate. |
| Release metadata checksum fails | Restore the original bound metadata; do not hand-edit a finalized card, schema, or manifest. |
| Observable logging policy is not exactly reproducible | Check the release source/runtime provenance, especially for the bundled sample. Generate and validate a fresh release in the active environment. |
| No rows for a requested split / too few training rows | Inspect `manifest.json`; use a larger release because small building samples can leave splits empty. |
| Memory pressure or poor worker utilization | Reduce batch size or workers; check `plan` to ensure enough shards exist for parallel work. |
| Publication target, visibility, or parent-commit conflict | Check the bound repository/branch and remote changes; resume the matching target or use a separate release. See the publishing guide. |

## Development and verification

Install all development and example dependencies, then run the repository checks:

```bash
python -m pip install -e ".[dev,hub,analysis,notebook]"
python -m pip check
python -m ruff check src tests examples scripts benchmarks
python -m ruff format --check src tests examples scripts benchmarks
python scripts/update_schema.py --check
python -m pytest -q
```

Tests cover independent thermal calculations, deterministic batching, int64 row
boundaries, split isolation, corrupted records and files, interruption recovery,
process leases, policy uncertainty, CLI behavior, notebook failures, and publication.
Publication tests use an in-memory Hub service that checks installed SDK signatures
and commit-parent semantics; they do not certify live account permissions or quotas.

Build and verify the distributable artifacts:

```bash
python -m build
python scripts/check_distribution.py
```

The distribution check expects one wheel and one source archive in `dist/`. It
checks their contents against the checkout, installs the wheel into a temporary
target, and generates, validates, and exactly replays a small dataset from that
installation. The package version is defined in
[src/thermoshift/_version.py](src/thermoshift/_version.py).

When changing column definitions, regenerate the reference with
`python scripts/update_schema.py`. Follow [CONTRIBUTING.md](CONTRIBUTING.md) for
scientific regression expectations and documentation maintenance. Report bugs or
propose changes through the repository's
[issue tracker](https://github.com/neuralsorcerer/thermoshift/issues), including the
command, package/runtime versions, relevant configuration, and error output.

## License

ThermoShift is licensed under the [MIT license](LICENSE).
