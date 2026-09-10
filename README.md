# ThermoShift

Synthetic cooling trajectories and counterfactual outcomes for machine learning.

ThermoShift generates hourly building-cooling decisions for temperature prediction,
action-effect estimation, policy evaluation, and sensor-state modeling. Each decision
contains observed context, a sampled cooling action, its logging probability, and
the resulting temperature, electricity use, cost, emissions, and comfort deviation.

A matching oracle table provides latent state and potential outcomes for all three
cooling actions. Records form ordered building trajectories and are written to
sharded Parquet files with building-level train, validation, and test splits.

## Installation

Requires Python 3.11 or later. From the project directory, install the package and
the extras used by the examples:

```bash
python -m pip install -e ".[hub,analysis]"
python -m thermoshift --help
```

Use `python -m pip install .` for the generator alone. Optional extras are `hub`
for Hugging Face publishing and loading, `analysis` for the regression baseline,
`notebook` for the walkthrough, and `dev` for development tools.

## Generate your first dataset

Initialize the configuration, generate the files, and validate the release:

```bash
python -m thermoshift init output/demo --rows 1000000 --seed 42
python -m thermoshift generate output/demo
python -m thermoshift validate output/demo
```

The row count is exact. With the default 168-hour episode length, the final
building's trajectory is shortened when necessary. One million decisions produce
one million `logged` rows and one million corresponding `oracle` rows, joined by
`row_id`.

Rerun `generate` to resume an interrupted job. Completed shards are verified;
missing or corrupt shards are regenerated. A run keeps its configuration and
generator fingerprint, so resume with the same source and environment. Use a new
output directory when changing the configuration or generator.

Commands write their final result as JSON to standard output. Generation progress
goes to standard error; use `generate --quiet` to suppress it.

## Load and explore

The repository includes a 50,003-decision sample. Load it directly with Hugging Face
Datasets:

```python
from datasets import load_dataset

records = load_dataset(
    "sample_dataset",
    name="logged",
    split="train",
    streaming=True,
    columns=["row_id", "building_id", "step", "obs_temp_c", "action", "y_next_temp_c"],
)

for record in records.take(3):
    print(record)
```

Replace `sample_dataset` with `output/demo` to use your generated release.

| Configuration | Contents | Typical use |
| --- | --- | --- |
| `logged` | Observed context, actions, propensities, factual targets, episode boundaries | Training and propensity-based estimation |
| `oracle` | Latent state, physical parameters, sensor labels, potential outcomes, best one-step action | Evaluation and explicitly selected oracle supervision |

`feature_roles.json` lists the inputs for each task. Use `policy_features` for
decision policies and `transition_features` for action-conditioned prediction.
Preserve building and step order for sequence models.

## Run the examples

```bash
python examples/train_baseline.py --data sample_dataset
python examples/evaluate_policy.py --data sample_dataset --split test
python examples/stream_hf.py sample_dataset --config logged --split train --limit 3
```

The regression example compares a gradient-boosted temperature model with
persistence on held-out buildings. The policy example computes IPS, SNIPS, and
oracle reward for a fixed threshold policy on logged states, with uncertainty
clustered by building.

For an interactive walkthrough:

```bash
python -m pip install -e ".[hub,analysis,notebook]"
python scripts/run_notebook.py
```

Open [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb) to inspect the saved
tables, split profiles, feature roles, and paired outcomes.

## Scale generation

Set the target size at initialization and use multiple workers to process shards:

```bash
python -m thermoshift init output/large --rows 1000000000 --seed 42
python -m thermoshift plan output/large
python -m thermoshift generate output/large --workers 8 --batch-buildings 512
python -m thermoshift validate output/large
```

`--batch-buildings` controls each worker's simulation batch. `--shard-buildings`
controls storage granularity and is fixed by `init`. Memory grows with worker
count, batch size, and episode length; file inventories grow with shard count.
Measure a pilot with [benchmarks/generation.py](benchmarks/generation.py) to choose
settings for your machine.

See [generation and operations](docs/generation.md) for configuration, distributed
ranks, recovery, and validation options.

## Dataset and package reference

Buildings are assigned to five splits: `train`, `validation`, `test`,
`test_heatwave`, and `test_sensor`. The two stress splits vary outdoor conditions,
equipment performance, outages, sensor drift, and dropout. The model uses an hourly
single-zone thermal equation and three cooling levels: off, half input, and full
input.

- [Model and parameter distributions](src/thermoshift/resources/DATASHEET.md)
- [Column reference](docs/schema.md)
- [Python API](docs/api.md)
- [Package architecture](docs/architecture.md)
- [Development and testing](CONTRIBUTING.md)

The generator and generated datasets use the [MIT license](LICENSE).
