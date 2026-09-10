---
pretty_name: ThermoShift
license: mit
task_categories:
- tabular-regression
- tabular-classification
- reinforcement-learning
tags:
- synthetic
- energy
- causal-inference
- counterfactual
- offline-rl
- distribution-shift
- time-series
configs:
- config_name: logged
  default: true
  data_files:
  - split: train
    path: data/logged/train/**/*.parquet
  - split: validation
    path: data/logged/validation/**/*.parquet
  - split: test
    path: data/logged/test/**/*.parquet
  - split: test_heatwave
    path: data/logged/test_heatwave/**/*.parquet
  - split: test_sensor
    path: data/logged/test_sensor/**/*.parquet
- config_name: oracle
  default: false
  data_files:
  - split: train
    path: data/oracle/train/**/*.parquet
  - split: validation
    path: data/oracle/validation/**/*.parquet
  - split: test
    path: data/oracle/test/**/*.parquet
  - split: test_heatwave
    path: data/oracle/test_heatwave/**/*.parquet
  - split: test_sensor
    path: data/oracle/test_sensor/**/*.parquet
---

# ThermoShift

Hourly synthetic building-cooling trajectories with observed decisions and paired
counterfactual outcomes.

The dataset contains three cooling actions, their logging probabilities, factual
outcomes, and an oracle table with latent state and potential outcomes for every
action. Buildings are assigned to train, validation, and test splits, including
heatwave and sensor-degradation conditions.

## Configurations

| Name | Contents |
| --- | --- |
| `logged` | Decision-time observations, cooling actions, propensities, factual targets, and trajectory boundaries |
| `oracle` | Latent temperature, physical parameters, sensor labels, three potential outcomes, and the best one-step action |

Join configurations using `row_id`; `building_id` and `step` identify the position
within each trajectory. `schema.json` describes every column, type, unit, and role.
`feature_roles.json` supplies the policy and transition-model feature lists.

## Loading

Use the repository ID shown on this dataset's page:

```python
from datasets import load_dataset

records = load_dataset(
    "YOUR_USERNAME/thermoshift",
    name="logged",
    split="train",
    streaming=True,
    columns=["row_id", "building_id", "step", "obs_temp_last_c", "action", "y_next_temp_c"],
)
print(next(iter(records)))
```

Set `revision` to a completed commit SHA when recording an experiment. Streaming
reads records as they are consumed. Keep trajectories ordered for sequential tasks
and match oracle labels by their identifiers.

## Tasks

- Predict next-hour temperature, energy, emissions, or comfort deviation.
- Estimate action effects using the paired outcomes as evaluation labels.
- Evaluate one-step policies on logged states with the supplied propensities.
- Reconstruct sensor state and identify drift or missing observations.
- Measure generalization across buildings and the two stress conditions.

Policy inputs come from `policy_features`. Transition predictors can also use the
logged action. Oracle fields provide evaluation labels or an explicitly selected
supervision source. One-step potential outcomes branch from the current factual
state; evaluating a different trajectory policy requires simulating its resulting
state sequence.

## Model and provenance

A single-zone thermal model advances each building by one hour. Cooling levels are
0%, 50%, and 100% of rated electrical input. Weather, occupancy, grid availability,
and sensing evolve over the trajectory. Electricity and comfort determine the
reward. [DATASHEET.md](DATASHEET.md) gives the equations and parameter distributions.

`run_config.json` records the configuration, generator fingerprint, and runtime
versions. `manifest.json` contains counts, sizes, and file checksums. The completion
marker binds the manifest and release metadata. `validation.json` records the most
recent saved validation status and its check scope.

## License

The dataset is distributed under the MIT license included in `LICENSE`.

## This release

50,003 unique hourly decision records from 298 synthetic buildings. Each decision has a matching row in the logged and oracle configurations, linked by row_id.

| Split | Decision records |
| --- | ---: |
| test | 4,032 |
| test_heatwave | 1,848 |
| test_sensor | 4,139 |
| train | 35,616 |
| validation | 4,368 |

Seed: `42`. Episode length: `168` hourly steps. Hidden-confounding mode: `False`. Config fingerprint: `11b11ad6e22f099a937182f50fd84774d7679576860cced9f2ab87d758161ef3`.

Compressed Parquet bytes (both configurations): `9916792`. Generation parameters and runtime versions are in `run_config.json`. Column roles and units are in `schema.json`.
