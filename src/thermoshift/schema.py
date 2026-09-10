# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Machine-readable units, feature roles, nullability, and Arrow schemas."""

import pyarrow as pa


def field(name, dtype, role, description, unit="1", nullable=False):
    return pa.field(
        name,
        dtype,
        nullable=nullable,
        metadata={
            "role": role,
            "description": description,
            "unit": unit,
        },
    )


ID_FIELDS = [
    field(
        "row_id", pa.int64(), "id", "building_id * episode_steps + step; unique within a release"
    ),
    field(
        "building_id",
        pa.int64(),
        "id",
        "Independent synthetic building, one trajectory per building",
    ),
    field("step", pa.int32(), "id", "Zero-based hour within the trajectory", "h"),
]
CONTEXT_FIELDS = [
    field("hour", pa.int8(), "feature", "Synthetic local hour, no geographic timezone", "h"),
    field("day_of_week", pa.int8(), "feature", "Synthetic weekday: Monday=0"),
    field("building_type", pa.int8(), "feature", "0=residential-like, 1=office-like"),
    field("floor_area_m2", pa.float32(), "feature", "Known floor area", "m2"),
    field(
        "hvac_capacity_kw",
        pa.float32(),
        "feature",
        "Rated electric cooling input at action 2",
        "kW",
    ),
    field(
        "setpoint_c",
        pa.float32(),
        "feature",
        "Constant preferred temperature for the episode",
        "degC",
    ),
    field(
        "outdoor_temp_c",
        pa.float32(),
        "feature",
        "Outdoor temperature held constant over the next hour",
        "degC",
    ),
    field("solar_w_m2", pa.float32(), "feature", "Synthetic solar irradiance", "W/m2"),
    field(
        "price_per_kwh", pa.float32(), "feature", "Known hourly illustrative tariff", "currency/kWh"
    ),
    field(
        "carbon_kg_per_kwh",
        pa.float32(),
        "feature",
        "Known illustrative grid carbon intensity",
        "kgCO2e/kWh",
    ),
    field(
        "occupancy_count",
        pa.int16(),
        "feature",
        "Idealized observable occupancy, constant this hour",
        "people",
    ),
    field(
        "grid_available", pa.bool_(), "feature", "Supply availability known at the decision time"
    ),
    field(
        "obs_temp_c",
        pa.float32(),
        "feature",
        "Current corrupted sensor measurement; null during dropout",
        "degC",
        True,
    ),
    field(
        "obs_temp_last_c",
        pa.float32(),
        "feature",
        "Latest available reading, initially setpoint if missing",
        "degC",
    ),
    field(
        "sensor_age_steps",
        pa.int32(),
        "feature",
        "Hours since available reading; increments from 1 on initial dropout",
        "h",
    ),
]
ACTION_FIELDS = [
    field("action", pa.int8(), "action", "0=off, 1=half input, 2=full input"),
    field(
        "propensity",
        pa.float64(),
        "propensity",
        "Float64 probability used by the actual logging policy for the selected action",
    ),
] + [
    field(
        f"p_action_{a}",
        pa.float64(),
        "propensity",
        f"Float64 logging probability used to sample action {a}",
    )
    for a in range(3)
]
TARGET_FIELDS = [
    field(
        "y_next_temp_c",
        pa.float32(),
        "target",
        "Simulated true temperature at the end of the hour",
        "degC",
    ),
    field(
        "y_energy_kwh",
        pa.float32(),
        "target",
        "Served background plus HVAC electricity over one hour",
        "kWh",
    ),
    field("y_cost", pa.float32(), "target", "Served electricity cost", "currency"),
    field(
        "y_carbon_kg",
        pa.float32(),
        "target",
        "Operational grid emissions for served electricity",
        "kgCO2e",
    ),
    field(
        "y_discomfort_c",
        pa.float32(),
        "target",
        "End-of-hour deviation beyond the setpoint +/-1 degC comfort band",
        "degC",
    ),
    field(
        "y_reward",
        pa.float32(),
        "target",
        "Negative weighted electricity, emissions and endpoint comfort cost",
        "currency-equivalent",
    ),
    field(
        "episode_end",
        pa.bool_(),
        "boundary",
        "Final stored step of the building trajectory",
    ),
]
ORACLE_FIELDS = [
    field("true_temp_c", pa.float32(), "oracle", "Latent temperature at decision time", "degC"),
    field(
        "conductance_kw_per_c", pa.float32(), "oracle", "Thermal conductance to ambient", "kW/degC"
    ),
    field("capacity_kwh_per_c", pa.float32(), "oracle", "Lumped thermal heat capacity", "kWh/degC"),
    field("base_load_kw", pa.float32(), "oracle", "Requested background electrical load", "kW"),
    field(
        "internal_heat_kw",
        pa.float32(),
        "oracle",
        "Occupant heat plus served background electric heat",
        "kW",
    ),
    field("solar_heat_kw", pa.float32(), "oracle", "Solar heat admitted to building", "kW"),
    field(
        "process_heat_kw", pa.float32(), "oracle", "Shared unmodeled hourly heat disturbance", "kW"
    ),
    field("cop", pa.float32(), "oracle", "Actual temperature-dependent coefficient of performance"),
    field(
        "sensor_bias_c", pa.float32(), "oracle", "Bias excluding white measurement noise", "degC"
    ),
    field(
        "sensor_fault",
        pa.bool_(),
        "oracle",
        "Dropout or absolute serialized sensor_bias_c greater than 0.75 degC",
    ),
]
for a in range(3):
    for name, unit in [
        ("next_temp_c", "degC"),
        ("energy_kwh", "kWh"),
        ("cost", "currency"),
        ("carbon_kg", "kgCO2e"),
        ("discomfort_c", "degC"),
        ("reward", "currency-equivalent"),
    ]:
        ORACLE_FIELDS.append(
            field(
                f"cf_{name}_{a}",
                pa.float32(),
                "oracle",
                f"One-step potential outcome under action {a}",
                unit,
            )
        )
ORACLE_FIELDS.append(
    field(
        "oracle_action",
        pa.int8(),
        "oracle",
        "Action maximizing serialized one-step reward; smallest index breaks ties",
    )
)

LOGGED_SCHEMA = pa.schema(ID_FIELDS + CONTEXT_FIELDS + ACTION_FIELDS + TARGET_FIELDS)
ORACLE_SCHEMA = pa.schema(ID_FIELDS + ORACLE_FIELDS)
FEATURES = [f.name for f in CONTEXT_FIELDS]
SCHEMAS = {"logged": LOGGED_SCHEMA, "oracle": ORACLE_SCHEMA}


def schema_document():
    return {
        kind: [
            {
                "name": f.name,
                "dtype": str(f.type),
                "nullable": f.nullable,
                **{k.decode(): v.decode() for k, v in f.metadata.items()},
            }
            for f in schema
        ]
        for kind, schema in SCHEMAS.items()
    }


def feature_roles():
    return {
        "policy_features": FEATURES.copy(),
        "transition_features": FEATURES + ["action"],
        "forbidden_policy_inputs": [f.name for f in LOGGED_SCHEMA if f.name not in FEATURES],
        "oracle_access": "evaluation or explicitly declared privileged training only",
    }
