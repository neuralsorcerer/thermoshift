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
