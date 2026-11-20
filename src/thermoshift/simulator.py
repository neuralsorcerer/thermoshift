# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Vectorized one-zone, one-hour RC simulator with shared-noise potential outcomes."""

from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

import numpy as np
import pyarrow as pa

from thermoshift.config import LEVELS, Config, integer
from thermoshift.random import normal, split_codes, uniform
from thermoshift.schema import SCHEMAS

_FLOAT64 = np.finfo(np.float64)
_CANCELLATION_RATIO = np.sqrt(_FLOAT64.eps)
_NEGLIGIBLE_DECAY_RATE = 1500
# A ratio of positive float64 inputs can be as small as 1e-632. Allow
# for that cancellation in 1 - decay and for cancellation in the sum.
_THERMAL_CONTEXT = Context(prec=1400, rounding=ROUND_HALF_EVEN, Emin=-9999, Emax=9999, traps=[])


def _thermal_input(value):
    if isinstance(value, np.ma.MaskedArray) and np.ma.is_masked(value):
        raise ValueError("thermal inputs must contain a real value at every position")
    array = np.asarray(value)
    if array.dtype.kind in "cmM" or (
        array.dtype.kind == "O"
        and any(
            isinstance(item, (complex, np.complexfloating, np.datetime64, np.timedelta64))
            for item in array.flat
        )
    ):
        raise ValueError("thermal inputs must be real numbers")
    try:
        return array.astype(np.float64, copy=False)
    except OverflowError as error:
        raise ValueError("thermal inputs must be finite in float64") from error


def _thermal_step_precise(temp, outdoor, conductance, capacity, heat_kw, cooling_kw):
    """Evaluate an exceptional element without platform-specific extended floats."""
    temp, outdoor, conductance, capacity, heat_kw, cooling_kw = (
        Decimal.from_float(float(value))
        for value in (temp, outdoor, conductance, capacity, heat_kw, cooling_kw)
    )
    with localcontext(_THERMAL_CONTEXT) as context:
        net_heat = heat_kw - cooling_kw
        if not conductance:
            return float(temp + net_heat / capacity)
        rate = conductance / capacity
        equilibrium = (outdoor * conductance + net_heat) / conductance
        # The transient is zero at equilibrium and too small to affect a
        # finite float64 result above the rate cutoff.
        if temp == equilibrium or rate >= _NEGLIGIBLE_DECAY_RATE:
            return float(equilibrium)
        # Keep digits to resolve 1 - decay when the rate is small. Opposite
        # signs need guard digits for cancellation. Most cases can finish
        # with these digits, so only deeper cancellation uses full precision.
        can_cancel = bool(temp and equilibrium) and temp.is_signed() != equilibrium.is_signed()
        guard_digits = 30 if can_cancel else 0
        first_precision = 35 + max(0, -rate.adjusted()) + guard_digits
        for precision in (first_precision, _THERMAL_CONTEXT.prec):
            context.prec = precision
            decay = (-rate).exp()
            interior = decay * temp
            exterior = (1 - decay) * equilibrium
            result = interior + exterior
            if not can_cancel or (
                result
                and result.adjusted()
                >= max(interior.adjusted(), exterior.adjusted()) - guard_digits
            ):
                break
        return float(result)


def thermal_step(temp, outdoor, conductance, capacity, heat_kw, cooling_kw):
    """Exact constant-forcing RC solution for dt=1 h, temperatures in degC.

    Inputs broadcast as real, finite float64 arrays. Conductance must be
    nonnegative and capacity must be positive. Zero conductance is the
    insulated-building limit.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        temp, outdoor, conductance, capacity, heat_kw, cooling_kw = (
            _thermal_input(value)
            for value in (temp, outdoor, conductance, capacity, heat_kw, cooling_kw)
        )
    if np.any(~np.isfinite(conductance) | (conductance < 0)) or np.any(
        ~np.isfinite(capacity) | (capacity <= 0)
    ):
        raise ValueError(
            "conductance must be finite and nonnegative and capacity must be finite and positive"
        )
    if any(np.any(~np.isfinite(value)) for value in (temp, outdoor, heat_kw, cooling_kw)):
        raise ValueError("temperature, outdoor temperature, heat and cooling must be finite")
    try:
        temp, outdoor, conductance, capacity, heat_kw, cooling_kw = np.broadcast_arrays(
            temp, outdoor, conductance, capacity, heat_kw, cooling_kw
        )
    except ValueError as error:
        raise ValueError("thermal inputs must have broadcast-compatible shapes") from error
    # Finite inputs can overflow intermediate coefficients or products.
    # Compute the usual vectorized solution, then retry affected elements.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rate = conductance / capacity
        alpha = -np.expm1(-rate)
        decay = np.subtract(1.0, alpha, out=np.empty_like(rate))
        # At higher rates, 1 - alpha loses significant digits or rounds to zero.
        np.exp(-rate, out=decay, where=rate >= 1.0)
        # The series covers zero conductance without dividing by it.
        small = rate < 1e-8
        heat_response = np.divide(alpha, conductance, out=np.zeros_like(rate), where=~small)
        np.divide(1.0 - 0.5 * rate, capacity, out=heat_response, where=small)
        indoor_term = decay * temp
        outdoor_term = alpha * outdoor
        heat_term = heat_kw * heat_response
        cooling_term = cooling_kw * heat_response
        forcing = heat_term - cooling_term
        result = indoor_term + outdoor_term + forcing
        largest_term = np.maximum(
            np.maximum(np.abs(indoor_term), np.abs(outdoor_term)),
            np.maximum(np.abs(heat_term), np.abs(cooling_term)),
        )
        # Retry if cancellation discards at least half the significant bits,
        # or if a subnormal coefficient or result may have rounded too early.
        retry = (
            ~np.isfinite(result)
            | (np.abs(result) < _CANCELLATION_RATIO * largest_term)
            | ((rate < _FLOAT64.tiny) & (conductance > 0))
            | ((decay < _FLOAT64.tiny) & (rate < _NEGLIGIBLE_DECAY_RATE) & (temp != 0))
            | (
                (np.abs(result) < _FLOAT64.tiny)
                & ((temp != 0) | ((conductance > 0) & (outdoor != 0)) | (heat_kw != cooling_kw))
            )
        )
    if np.any(retry):
        result = np.asarray(result)
        inputs = (temp, outdoor, conductance, capacity, heat_kw, cooling_kw)
        corrected = [
            _thermal_step_precise(*values)
            for values in zip(*(value[retry] for value in inputs), strict=True)
        ]
        if not np.all(np.isfinite(corrected)):
            raise ValueError("thermal inputs produce a nonfinite temperature")
        result[retry] = corrected
        return result[()]
    return result


def _buffer(schema, n, steps):
    types = {
        pa.int64(): np.int64,
        pa.int32(): np.int32,
        pa.int16(): np.int16,
        pa.int8(): np.int8,
        pa.float32(): np.float32,
        pa.float64(): np.float64,
        pa.bool_(): bool,
    }
    return {f.name: np.empty((n, steps), dtype=types[f.type]) for f in schema}


def simulate(
    config: Config, start_building: int, stop_building: int
) -> tuple[pa.Table, pa.Table, np.ndarray]:
    """Return (logged Arrow table, oracle Arrow table, split codes per row).

    Memory is O((stop-start)*episode_steps), never O(config.rows). The caller
    chooses a bounded building batch. All complete trajectories are building-major.
    Only the dataset's final trajectory may be truncated to obtain an exact row count.
    """
    if not isinstance(config, Config):
        raise TypeError("config must be a Config instance")
    integer(start_building, "start_building", 0, config.buildings - 1)
    integer(stop_building, "stop_building", start_building + 1, config.buildings)
    ids = np.arange(start_building, stop_building, dtype=np.int64)
    n, steps, seed = len(ids), config.episode_steps, config.seed

    def u(stream, t=0):
        return uniform(ids, seed, stream, t)

    def z(stream, t=0):
        return normal(ids, seed, stream, t)

    def observed(value):
        # The logging policy receives precisely the context serialized for learners.
        return np.asarray(value, dtype=np.float32).astype(np.float64)

    split = split_codes(ids, seed)
    hot, bad_sensor = split == 3, split == 4
    area = 50 + 250 * u(1)
    building_type = (u(2) < 0.35).astype(np.int8)
    conductance = 0.0025 * area * (0.65 + 0.70 * u(3))
    capacity = 0.040 * area * (0.7 + 0.60 * u(4))
    hvac = 0.025 * area * (0.8 + 0.4 * u(5))
    aperture = area * (0.035 + 0.065 * u(6))
    cop_nominal = 3.2 + 0.8 * u(7)
    setpoint = 23 + 2 * u(8)
    population = (1 + np.floor(area / 35 * u(9))).astype(np.int16)
    weekday0 = np.floor(7 * u(10)).astype(np.int32)
    weather_offset = 2.0 * z(11) + 8.0 * hot
    sensor_initial_bias = 0.12 * z(12)
    sensor_drift = z(13) * np.where(bad_sensor, 3.0, 0.20)
    temp = setpoint + z(14)
    last_obs = setpoint.copy()
    age = np.zeros(n, dtype=np.int32)
    weather_noise = np.zeros(n)
    dropout = np.zeros(n, dtype=bool)
    outage = np.zeros(n, dtype=bool)
    logged = _buffer(SCHEMAS["logged"], n, steps)
    oracle = _buffer(SCHEMAS["oracle"], n, steps)
    rows = np.arange(n)
    levels = np.asarray(LEVELS)
    first_row_ids = ids * steps
    episode_lengths = np.minimum(steps, config.rows - first_row_ids)

    for t in range(steps):
        hour = t % 24
        weekday = (weekday0 + t // 24) % 7
        weather_noise = 0.92 * weather_noise + 0.35 * z(101, t)
        outdoor = 29 + 5 * np.sin(2 * np.pi * (hour - 9) / 24) + weather_offset + weather_noise
        solar = np.maximum(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0.0
        irradiance = solar * (550 + 350 * u(102, t))
        office_active = (weekday < 5) & (8 <= hour) & (hour < 18)
        residential_active = (hour < 9) | (hour >= 17) | (weekday >= 5)
        presence = np.where(building_type == 1, office_active, residential_active)
        occupancy = np.where(u(103, t) < np.where(presence, 0.92, 0.18), population, 0).astype(
            np.int16
        )
        base_load = area * 0.003 * (0.35 + 0.65 * occupancy / population)
        outage_p = np.where(outage, 0.55, np.where(hot, 0.012, 0.002))
        outage = u(104, t) < outage_p
        grid = ~outage
        price = np.clip(
            0.10
            + 0.14 * (16 <= hour < 21)
            + 0.003 * np.maximum(outdoor - 30, 0)
            + 0.01 * z(105, t),
            0.04,
            0.65,
        )
        carbon = np.clip(
            0.42 - 0.16 * solar + 0.10 * (17 <= hour < 22) + 0.025 * z(106, t), 0.10, 0.85
        )
        dropout_p = np.where(
            dropout, np.where(bad_sensor, 0.85, 0.50), np.where(bad_sensor, 0.10, 0.008)
        )
        dropout = u(107, t) < dropout_p
        bias = sensor_initial_bias + sensor_drift * t / (steps - 1)
        reading = temp + bias + 0.15 * z(108, t)
        last_obs = np.where(dropout, last_obs, reading)
        age = np.where(dropout, age + 1, 0).astype(np.int32)

        # History-based logging policy mixed with uniform exploration.
        control_temp = observed(last_obs)
        if config.hidden_confounding:
            control_temp += 0.8 * (temp - control_temp)
        demand = np.clip(
            (control_temp - observed(setpoint) + 0.15 * (observed(outdoor) - observed(setpoint)))
            / 3.0,
            0,
            1,
        )
        logits = -5 * (levels[None, :] - demand[:, None]) ** 2
        logits -= 0.7 * observed(price)[:, None] * observed(hvac)[:, None] * levels
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        probs = (1 - config.exploration) * probs + config.exploration / 3
        action = np.sum(u(109, t)[:, None] > probs.cumsum(axis=1)[:, :2], axis=1).astype(np.int8)

        cop = np.clip(cop_nominal - 0.045 * np.maximum(outdoor - 25, 0), 1.2, 5.0)
        cop *= np.where(hot, 0.85, 1.0)
        internal = 0.12 * occupancy + grid * base_load
        solar_heat = irradiance / 1000 * aperture
        process_heat = 0.06 * z(110, t)
        cooling = hvac[:, None] * levels * grid[:, None] * cop[:, None]
        next_temp = thermal_step(
            temp[:, None],
            outdoor[:, None],
            conductance[:, None],
            capacity[:, None],
            (internal + solar_heat + process_heat)[:, None],
            cooling,
        )
        energy = grid[:, None] * (base_load[:, None] + hvac[:, None] * levels)
        cost = price[:, None] * energy
        emissions = carbon[:, None] * energy
        discomfort = np.maximum(np.abs(next_temp - setpoint[:, None]) - 1.0, 0)
        comfort_occupancy_weight = np.where(occupancy > 0, 1.0, 0.1)
        reward = -(
            cost
            + config.carbon_weight * emissions
            + config.comfort_weight * comfort_occupancy_weight[:, None] * discomfort**2
        )

        # Unexported padding in the final trajectory must not overflow int64.
        row_ids = first_row_ids + np.minimum(t, episode_lengths - 1)
        values = {
            "row_id": row_ids,
            "building_id": ids,
            "step": t,
            "hour": hour,
            "day_of_week": weekday,
            "building_type": building_type,
            "floor_area_m2": area,
            "hvac_capacity_kw": hvac,
            "setpoint_c": setpoint,
            "outdoor_temp_c": outdoor,
            "solar_w_m2": irradiance,
            "price_per_kwh": price,
            "carbon_kg_per_kwh": carbon,
            "occupancy_count": occupancy,
            "grid_available": grid,
            "obs_temp_c": np.where(dropout, np.nan, reading),
            "obs_temp_last_c": last_obs,
            "sensor_age_steps": age,
            "action": action,
            "propensity": probs[rows, action],
            "episode_end": t == episode_lengths - 1,
        }
        for a in range(3):
            values[f"p_action_{a}"] = probs[:, a]
        outcomes = {
            "next_temp_c": next_temp,
            "energy_kwh": energy,
            "cost": cost,
            "carbon_kg": emissions,
            "discomfort_c": discomfort,
            "reward": reward,
        }
        for name, result in outcomes.items():
            values[f"y_{name}"] = result[rows, action]
        for name, value in values.items():
            logged[name][:, t] = value

        truths = {
            "row_id": row_ids,
            "building_id": ids,
            "step": t,
            "true_temp_c": temp,
            "conductance_kw_per_c": conductance,
            "capacity_kwh_per_c": capacity,
            "base_load_kw": base_load,
            "internal_heat_kw": internal,
            "solar_heat_kw": solar_heat,
            "process_heat_kw": process_heat,
            "cop": cop,
            "sensor_bias_c": bias,
            "sensor_fault": dropout | (np.abs(bias.astype(np.float32)) > 0.75),
            "oracle_action": np.argmax(reward.astype(np.float32), axis=1),
        }
        for a in range(3):
            for name, result in outcomes.items():
                truths[f"cf_{name}_{a}"] = result[:, a]
        for name, value in truths.items():
            oracle[name][:, t] = value
        temp = next_temp[rows, action]

    keep = min(n * steps, config.rows - start_building * steps)

    def to_arrow(buffer, schema):
        arrays = []
        for f in schema:
            data = buffer[f.name].reshape(-1)[:keep]
            arrays.append(pa.array(data, type=f.type, mask=np.isnan(data) if f.nullable else None))
        return pa.Table.from_arrays(arrays, schema=schema)

    return (
        to_arrow(logged, SCHEMAS["logged"]),
        to_arrow(oracle, SCHEMAS["oracle"]),
        np.repeat(split, steps)[:keep],
    )
