# ThermoShift

**Synthetic building-cooling trajectories and paired counterfactual outcomes
for machine learning.**

ThermoShift generates hourly cooling decisions with observed context, actions,
logging probabilities, and factual outcomes. A paired oracle table supplies
latent state and potential outcomes for all three cooling actions. Releases use
sharded Parquet files, deterministic building splits, checksums, resumable
generation, and optional Hugging Face publication.

## Get started

Use Python 3.11 or later. From a source checkout:

```bash
git clone https://github.com/neuralsorcerer/thermoshift.git
cd thermoshift
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[hub,analysis]"

python -m thermoshift init output/demo --rows 10000 --seed 42
python -m thermoshift generate output/demo
python -m thermoshift validate output/demo
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.
The release contains exactly 10,000 decisions: 10,000 `logged` rows and matching
`oracle` rows joined by `row_id`. A successful full validation reports
`"status": "passed"` and `"scope": "full"`.

```python
from datasets import load_dataset

records = load_dataset("output/demo", name="logged", split="train", streaming=True)
print(next(iter(records)))
```

## Choose a guide

- [Generation and operations](generation.md): configuration, scaling, ranks,
  recovery, and validation.
- [Python API](api.md): generation, verified paired readers, simulation, and
  reference experiments, schema helpers, publication, notebook execution, and
  the command-line result contracts.
- [Column reference](schema.md): field types, units, nullability, and feature roles.
- [Model and data reference](model.md): thermal equations, parameter distributions,
  sensors, propensities, and stress conditions.
- [Publishing and consumption](publishing.md): Hub release verification and streaming.
- [Package architecture](architecture.md): module responsibilities and release lifecycle.
- [Development and documentation](development.md): local checks, Sphinx builds,
  and CI deployment.

## Interpret the data

Buildings belong to disjoint train, validation, and test splits, including heatwave
and sensor-degradation tests. Keep trajectories ordered for sequence models and
select observed inputs using `feature_roles.json` to avoid target leakage.

The thermal model is an idealized research model. Counterfactual branches represent
one hour from the current logged state; they do not establish the long-run value
of a replacement policy or real-world controller performance. See the
[model reference](model.md) for its assumptions.

## Project resources

The [repository README](https://github.com/neuralsorcerer/thermoshift/blob/main/README.md)
contains the complete project overview and troubleshooting guide. Use the
[issue tracker](https://github.com/neuralsorcerer/thermoshift/issues) for bugs and
feature requests. The generator and generated datasets use the
[MIT license](https://github.com/neuralsorcerer/thermoshift/blob/main/LICENSE).

```{toctree}
:hidden:
:maxdepth: 2

generation
api
schema
model
publishing
architecture
development
```
