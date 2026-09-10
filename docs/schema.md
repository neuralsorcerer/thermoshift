# Column reference

[Project overview](../README.md) · [Python API](api.md)

Columns are defined in `src/thermoshift/schema.py`. Each configuration has one row per
decision, joined by `row_id`. The `role` field identifies how a column is used.

## logged

| Column | Type | Unit | Role | Nullable | Description |
| --- | --- | --- | --- | --- | --- |
| `row_id` | int64 | 1 | id | no | building_id * episode_steps + step; unique within a release |
| `building_id` | int64 | 1 | id | no | Independent synthetic building, one trajectory per building |
| `step` | int32 | h | id | no | Zero-based hour within the trajectory |
| `hour` | int8 | h | feature | no | Synthetic local hour, no geographic timezone |
| `day_of_week` | int8 | 1 | feature | no | Synthetic weekday: Monday=0 |
| `building_type` | int8 | 1 | feature | no | 0=residential-like, 1=office-like |
| `floor_area_m2` | float32 | m2 | feature | no | Known floor area |
| `hvac_capacity_kw` | float32 | kW | feature | no | Rated electric cooling input at action 2 |
| `setpoint_c` | float32 | degC | feature | no | Constant preferred temperature for the episode |
| `outdoor_temp_c` | float32 | degC | feature | no | Outdoor temperature held constant over the next hour |
| `solar_w_m2` | float32 | W/m2 | feature | no | Synthetic solar irradiance |
| `price_per_kwh` | float32 | currency/kWh | feature | no | Known hourly illustrative tariff |
| `carbon_kg_per_kwh` | float32 | kgCO2e/kWh | feature | no | Known illustrative grid carbon intensity |
| `occupancy_count` | int16 | people | feature | no | Idealized observable occupancy, constant this hour |
| `grid_available` | bool | 1 | feature | no | Supply availability known at the decision time |
| `obs_temp_c` | float32 | degC | feature | yes | Current corrupted sensor measurement; null during dropout |
| `obs_temp_last_c` | float32 | degC | feature | no | Latest available reading, initially setpoint if missing |
| `sensor_age_steps` | int32 | h | feature | no | Hours since available reading; increments from 1 on initial dropout |
| `action` | int8 | 1 | action | no | 0=off, 1=half input, 2=full input |
| `propensity` | float64 | 1 | propensity | no | Float64 probability used by the actual logging policy for the selected action |
| `p_action_0` | float64 | 1 | propensity | no | Float64 logging probability used to sample action 0 |
| `p_action_1` | float64 | 1 | propensity | no | Float64 logging probability used to sample action 1 |
| `p_action_2` | float64 | 1 | propensity | no | Float64 logging probability used to sample action 2 |
| `y_next_temp_c` | float32 | degC | target | no | Simulated true temperature at the end of the hour |
| `y_energy_kwh` | float32 | kWh | target | no | Served background plus HVAC electricity over one hour |
| `y_cost` | float32 | currency | target | no | Served electricity cost |
| `y_carbon_kg` | float32 | kgCO2e | target | no | Operational grid emissions for served electricity |
| `y_discomfort_c` | float32 | degC | target | no | End-of-hour deviation beyond the setpoint +/-1 degC comfort band |
| `y_reward` | float32 | currency-equivalent | target | no | Negative weighted electricity, emissions and endpoint comfort cost |
| `episode_end` | bool | 1 | boundary | no | Final stored step of the building trajectory |

## oracle

| Column | Type | Unit | Role | Nullable | Description |
| --- | --- | --- | --- | --- | --- |
| `row_id` | int64 | 1 | id | no | building_id * episode_steps + step; unique within a release |
| `building_id` | int64 | 1 | id | no | Independent synthetic building, one trajectory per building |
| `step` | int32 | h | id | no | Zero-based hour within the trajectory |
| `true_temp_c` | float32 | degC | oracle | no | Latent temperature at decision time |
| `conductance_kw_per_c` | float32 | kW/degC | oracle | no | Thermal conductance to ambient |
| `capacity_kwh_per_c` | float32 | kWh/degC | oracle | no | Lumped thermal heat capacity |
| `base_load_kw` | float32 | kW | oracle | no | Requested background electrical load |
| `internal_heat_kw` | float32 | kW | oracle | no | Occupant heat plus served background electric heat |
| `solar_heat_kw` | float32 | kW | oracle | no | Solar heat admitted to building |
| `process_heat_kw` | float32 | kW | oracle | no | Shared unmodeled hourly heat disturbance |
| `cop` | float32 | 1 | oracle | no | Actual temperature-dependent coefficient of performance |
| `sensor_bias_c` | float32 | degC | oracle | no | Bias excluding white measurement noise |
| `sensor_fault` | bool | 1 | oracle | no | Dropout or absolute serialized sensor_bias_c greater than 0.75 degC |
| `cf_next_temp_c_0` | float32 | degC | oracle | no | One-step potential outcome under action 0 |
| `cf_energy_kwh_0` | float32 | kWh | oracle | no | One-step potential outcome under action 0 |
| `cf_cost_0` | float32 | currency | oracle | no | One-step potential outcome under action 0 |
| `cf_carbon_kg_0` | float32 | kgCO2e | oracle | no | One-step potential outcome under action 0 |
| `cf_discomfort_c_0` | float32 | degC | oracle | no | One-step potential outcome under action 0 |
| `cf_reward_0` | float32 | currency-equivalent | oracle | no | One-step potential outcome under action 0 |
| `cf_next_temp_c_1` | float32 | degC | oracle | no | One-step potential outcome under action 1 |
| `cf_energy_kwh_1` | float32 | kWh | oracle | no | One-step potential outcome under action 1 |
| `cf_cost_1` | float32 | currency | oracle | no | One-step potential outcome under action 1 |
| `cf_carbon_kg_1` | float32 | kgCO2e | oracle | no | One-step potential outcome under action 1 |
| `cf_discomfort_c_1` | float32 | degC | oracle | no | One-step potential outcome under action 1 |
| `cf_reward_1` | float32 | currency-equivalent | oracle | no | One-step potential outcome under action 1 |
| `cf_next_temp_c_2` | float32 | degC | oracle | no | One-step potential outcome under action 2 |
| `cf_energy_kwh_2` | float32 | kWh | oracle | no | One-step potential outcome under action 2 |
| `cf_cost_2` | float32 | currency | oracle | no | One-step potential outcome under action 2 |
| `cf_carbon_kg_2` | float32 | kgCO2e | oracle | no | One-step potential outcome under action 2 |
| `cf_discomfort_c_2` | float32 | degC | oracle | no | One-step potential outcome under action 2 |
| `cf_reward_2` | float32 | currency-equivalent | oracle | no | One-step potential outcome under action 2 |
| `oracle_action` | int8 | 1 | oracle | no | Action maximizing serialized one-step reward; smallest index breaks ties |
