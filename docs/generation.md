# Generation and operations

[Project overview](index.md) · [Python API](api.md) · [Publishing](publishing.md)

## Configure a run

`init` writes an immutable generation plan. The main parameters are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--rows` | 1,000,000 | Exact number of hourly decisions |
| `--seed` | 42 | Seed for indexed random draws and building splits |
| `--episode-steps` | 168 | Hours per building trajectory, from 2 to 8,760 |
| `--shard-buildings` | 16,384 | Buildings assigned to each storage shard |
| `--exploration` | 0.15 | Uniform component of the action policy, in (0,1] |
| `--hidden-confounding` | Off | Include latent temperature in action assignment |
| `--comfort-weight` | 0.30 | Weight of squared endpoint comfort deviation |
| `--carbon-weight` | 0.05 | Weight of operational electricity emissions |

```bash
python -m thermoshift init output/experiment \
  --rows 2000000 --seed 17 --episode-steps 168 --shard-buildings 4096
python -m thermoshift plan output/experiment --batch-buildings 256
python -m thermoshift generate output/experiment --workers 4 --batch-buildings 256
python -m thermoshift validate output/experiment
```

Each shard produces a Parquet file for every nonempty split in each configuration.
Directories group 1,000 shard IDs. The final manifest records the exact layout.

## Choose batch and shard sizes

Batch size controls the number of buildings simulated together. A batch contains
up to `batch_buildings * episode_steps` transitions. Array allocation, Arrow
conversion, partition filtering, and compression contribute to worker memory.
Reduce `--batch-buildings` or worker count when memory is constrained.

Shard size controls file size and the amount of work rebuilt after interruption.
It is fixed at initialization. Smaller shards expose more independent jobs and
produce more files. Workers are capped by the number of shards assigned to the
current rank. A single assigned shard runs in the calling process.

Use the [benchmark script](https://github.com/neuralsorcerer/thermoshift/blob/main/benchmarks/README.md) to measure a representative run.
For one billion decisions with default episode and shard sizes, the configuration
contains 5,952,381 buildings and 364 storage shards, with at most 3,640 Parquet files.

## Resume and recover

Repeat the generation command with the same source, runtime, and output directory.
Shard checksums determine which work is complete. A shard with missing or changed
files is regenerated. Changing worker count or batch size preserves numeric records;
changing row-group boundaries can change the compressed file bytes.

A shard becomes committed after its Parquet files are closed, flushed, renamed,
and described by an atomic state file. Finalization checks every shard and writes
the release manifest, dataset card, and completion marker.
Refinalizing clears the previous completion marker and validation report before
checking shards. If finalization fails, finish recovery and validate again before
consuming the release.

| Situation | Action |
| --- | --- |
| Generation was interrupted | Rerun `generate` for the same plan |
| A file checksum fails | Run `generate`, then validate the rebuilt release |
| Finalization reports an incomplete shard | Finish all assigned generation ranks, then finalize |
| Source or runtime differs from initialization | Restore that environment or initialize a new directory |
| Dataset is busy | Finish or close the operation holding its dataset lease, then retry |
| Configuration has changed | Initialize a separate output directory |

## Run multiple ranks

Ranks share a filesystem that supports advisory locks and atomic same-directory
renames. Initialize the directory once, then run each rank separately. Shard `i`
belongs to rank `i % world_size`.

```bash
python -m thermoshift init /shared/thermoshift --rows 1000000000

# Run each command on its assigned machine or job.
python -m thermoshift generate /shared/thermoshift --world-size 4 --rank 0 --workers 8
python -m thermoshift generate /shared/thermoshift --world-size 4 --rank 1 --workers 8
python -m thermoshift generate /shared/thermoshift --world-size 4 --rank 2 --workers 8
python -m thermoshift generate /shared/thermoshift --world-size 4 --rank 3 --workers 8

# Run after every rank finishes.
python -m thermoshift finalize /shared/thermoshift
python -m thermoshift validate /shared/thermoshift
```

Generation ranks use shared lifecycle leases and separate shard locks. Finalizing,
validating, and publishing require exclusive leases. Child processes hold their own
leases while writing. All programs modifying an active dataset must follow this
locking protocol. Generate on the shared filesystem before uploading to a remote
storage service.

## Validate a release

```bash
# File integrity and every record-level check.
python -m thermoshift validate output/experiment

# File hashes, metadata, schemas and row counts.
python -m thermoshift validate output/experiment --metadata-only

# Include exact regeneration of each building.
python -m thermoshift validate output/experiment --replay

# Print the result while preserving the existing report file.
python -m thermoshift validate output/experiment --no-report
```

Full checks cover identity, split membership, trajectory continuity, sensor history,
logging probabilities, factual/oracle alignment, thermal and electricity equations,
reward accounting, and one-step action labels. Numeric equations are compared with
serialization tolerances; copied values and identifiers are checked exactly.

The saved report uses `running`, `passed`, `failed`, or `interrupted` status. Its
scope identifies the checks performed. Replay requires the initialized generator
and runtime and adds a building-by-building comparison. A process stopped before
cleanup may leave `running`; rerun validation to refresh the report.

## Generated files

| Path | Purpose |
| --- | --- |
| `data/logged/<split>/<bucket>/part-*.parquet` | Observations, logging information and factual outcomes |
| `data/oracle/<split>/<bucket>/part-*.parquet` | Latent variables and potential outcomes |
| `run_config.json` | Configuration and generator/runtime fingerprints |
| `schema.json`, `feature_roles.json` | Column definitions and permitted feature lists |
| `manifest.json` | Exact file inventory, counts, sizes and SHA-256 hashes |
| `README.md`, `DATASHEET.md`, `LICENSE` | Dataset card, model reference and license |
| `validation.json` | Saved validation result |
| `_SUCCESS.json` | Completed-release metadata binding |
| `_state/` | Committed shard state and local publication binding |

CLI commands emit a JSON result to stdout on success and errors or progress to
stderr. `--version` prints the installed version without a subcommand. Parser
errors are handled by `argparse` and do not emit JSON. Handled `ValueError` and
`OSError` failures return status 2; interruption returns 130.

The successful result shapes are documented in the [Python API](api.md): `init`
returns plan metadata, `plan` returns row/shard/batch counts, `generate` returns
written/resumed counts and elapsed seconds, `finalize` returns compact manifest
counts, and `validate` returns its validation report. `generate` automatically
finalizes only single-rank runs; multi-rank runs require an explicit `finalize`.
