# Model and data reference

ThermoShift simulates one cooling decision per building-hour. Each building starts
with its own physical parameters and runs for the configured episode length,
which defaults to 168 hours. The final trajectory is shortened to obtain the exact
requested dataset size.

Each decision appears in two configurations: `logged` contains observed context
and factual outcomes; `oracle` contains latent variables and potential outcomes.
`row_id = building_id * episode_steps + step` joins the two views.

## State transition

Let indoor and outdoor temperature be $T$ and $T_o$, thermal conductance be
$G$ in kW/°C, and heat capacity be $C$ in kWh/°C. Let $Q$ be internal,
solar, and residual heat in kW, $P$ the rated electrical cooling input, $\eta$
the coefficient of performance, and $g$ grid availability.

An action selects $a\in\{0,0.5,1\}$. Inputs remain constant during the hour:

$$
C\frac{dT}{dt}=G(T_o-T)+Q-\eta Pag.
$$

For a one-hour interval, define $\alpha=1-e^{-G/C}$. The transition is

$$
T_{t+1}(a)=T_t+\alpha\left[T_o-T_t+\frac{Q-\eta Pag}{G}\right].
$$

The implementation evaluates the exponential coefficient with `expm1`. All three
potential outcomes use the same current temperature and hourly disturbances. The
sampled action selects the state that advances the factual trajectory.

## Electricity, comfort, and reward

For requested background electrical load $B$, served energy over the hour is

$$
E(a)=g(B+Pa)\times1\text{ hour}.
$$

Internal heat is $0.12$ kW per occupant plus the served background electrical
load. Solar heat is irradiance divided by 1,000 and multiplied by the building's
effective aperture. A grid outage sets served electricity and HVAC cooling to zero.

$$
\begin{aligned}
\operatorname{cost}(a)&=\operatorname{price}\,E(a),\\
\operatorname{carbon}(a)&=\operatorname{intensity}\,E(a),\\
D(a)&=\max\bigl(|T_{t+1}(a)-T_{\rm set}|-1,0\bigr),\\
R(a)&=-\operatorname{cost}(a)-w_c\operatorname{carbon}(a)-w_d uD(a)^2.
\end{aligned}
$$

The occupancy weight $u$ is 1 when occupied and 0.1 otherwise. Default reward
weights are `carbon_weight=0.05` and `comfort_weight=0.30`. Prices use a common
currency unit. Comfort is measured at the end of the hour.

## Building and disturbance distributions

The following are the generator's parameter distributions. $U$ denotes a
uniform draw on (0,1), and $Z$ a standard normal draw. Draws are indexed by seed,
building, stream, and time step.

| Parameter | Rule |
| --- | --- |
| Floor area $A$ | $50+250U$ m² |
| Building type | Office-like with probability 0.35; residential-like otherwise |
| Conductance $G$ | $0.0025A(0.65+0.70U)$ kW/°C |
| Heat capacity $C$ | $0.040A(0.70+0.60U)$ kWh/°C |
| Rated cooling input $P$ | $0.025A(0.8+0.4U)$ kW |
| Effective solar aperture | $A(0.035+0.065U)$ m² |
| Nominal COP | $3.2+0.8U$ |
| Setpoint | $23+2U$ °C |
| Initial temperature | Setpoint + $Z$ °C |
| Occupant capacity | $1+\lfloor A U/35\rfloor$ people |
| Initial weekday | Uniform integer from 0 through 6 |
| Building weather offset | $2Z$ °C |
| Residual heat | $0.06Z$ kW per hour |
| Initial sensor bias | $0.12Z$ °C |
| Total sensor drift | $0.20Z$ °C over the episode |
| Measurement noise | $0.15Z$ °C |

Outdoor temperature is

$$
T_o=29+5\sin\left(2\pi(h-9)/24\right)+\operatorname{offset}+w_t,
\quad w_t=0.92w_{t-1}+0.35Z_t,
$$

with $w_{-1}=0$. Daylight irradiance is
$\max(0,\sin(\pi(h-6)/12))(550+350U)$ W/m² between 06:00 and 18:00,
and zero at other hours.

Offices are active on weekdays from 08:00 to 18:00. Residential buildings are
active before 09:00, from 17:00 onward, and throughout weekends. Occupancy equals
the building's capacity with probability 0.92 during active hours and 0.18
otherwise; remaining hours have zero occupants. Background load is
$0.003A(0.35+0.65\,\operatorname{occupancy}/\operatorname{capacity})$ kW.

Actual COP is
$\operatorname{clip}(\operatorname{COP}_{\rm nominal}-0.045\max(T_o-25,0),1.2,5)$.
The tariff and carbon intensity follow:

```text
price = clip(0.10 + 0.14 I(16 <= hour < 21)
             + 0.003 max(outdoor - 30, 0) + 0.01 Z, 0.04, 0.65)
carbon_intensity = clip(0.42 - 0.16 daylight_sine
                        + 0.10 I(17 <= hour < 22) + 0.025 Z, 0.10, 0.85)
```

## Outages and sensing

Before each decision, the outage state transitions from available to unavailable
with probability 0.002, or remains unavailable with probability 0.55. The sensor
enters dropout with probability 0.008, or remains in dropout with probability 0.50.
Both chains are initialized in their available state before the first transition.

Sensor bias includes an initial offset and a linear drift across the episode.
Dropout is stored as an Arrow null in `obs_temp_c`. The policy uses
`obs_temp_last_c`, which retains the latest available reading and initially falls
back to the setpoint. `sensor_age_steps` is zero for available readings and
increments during dropout. `sensor_fault` is true during dropout or when the
absolute serialized bias exceeds 0.75 °C.

## Logging policy

The action policy uses published observations promoted from float32 to float64.
For the latest available sensor temperature $T_{\rm obs}$, define

$$
\begin{aligned}
d&=\operatorname{clip}\bigl((T_{\rm obs}-T_{\rm set}+0.15(T_o-T_{\rm set}))/3,0,1\bigr),\\
\ell(a)&=-5(a-d)^2-0.7\operatorname{price}\,Pa,\\
p(a)&=(1-\epsilon)\operatorname{softmax}(\ell)_a+\epsilon/3.
\end{aligned}
$$

The exploration parameter defaults to 0.15. A separate indexed uniform draw
samples the action. `p_action_0`, `p_action_1`, and `p_action_2` store the float64
probabilities used for sampling; `propensity` contains the selected probability.

With `hidden_confounding=True`, the policy substitutes
$T_{\rm obs}+0.8(T_{\rm true}-T_{\rm obs})$ for the observed control temperature.
The recorded propensities then condition on that hidden state. Use the supplied
probabilities when evaluating the implemented assignment process.

## Splits and stress conditions

A hash of building ID and seed assigns every building to one split. Default hash
bucket proportions are 70% train, 10% validation, 10% test, 5% heatwave test, and
5% sensor test. The manifest records the realized row counts.

| Split | Generating conditions |
| --- | --- |
| `train`, `validation`, `test` | Common parameter distributions, disjoint buildings |
| `test_heatwave` | Outdoor offset +8 °C, actual COP multiplied by 0.85, outage-entry probability 0.012 |
| `test_sensor` | Total drift $3Z$ °C, dropout-entry probability 0.10, dropout persistence 0.85 |

## Record semantics

Physics and reward calculations use float64. Measured features and outcome columns
are serialized as float32; action probabilities remain float64. The one-step
`oracle_action` maximizes serialized counterfactual reward, with the smallest action
index resolving ties.

`episode_end` marks the final stored step. Treat it as trajectory truncation when
constructing a sequential-control task. Counterfactual branches start from the
current factual state, so a replacement policy's trajectory is obtained by
resimulating its subsequent states.

Building trajectories are the grouping unit for uncertainty estimates and data
splitting. With the same seed, episode length, and model settings, extending the
row count preserves existing decision values. `episode_end` changes when a
previously truncated building is extended. Changing episode duration also changes
the per-hour sensor-drift schedule.
