# The model

## Network

Each zone has two nodes:

- **air** (air + furniture): capacity `volume × 1200 J/m³K × air_furniture_factor`,
- **mass** (walls, floors): capacity `area × C_m_per_m2`, coupled to the air by `area × H_am_per_m2`.

A floor-heated zone gets a **slab** node between the water and the room; a hot-water tank is one
more node. Envelope elements connect the air node (or the slab, for a heated floor) to three
boundaries: outdoor air, ground (`T_ground`) and neighbour (`T_neighbour`). Internal links connect
air nodes of different zones.

    C dx/dt = −G x + Gb T_boundary + q(t)

The linear part is discretised exactly (matrix exponential, zero-order hold); the heat inputs q
are evaluated each step:

- **radiators**, ntu model: water at `T_flow` flows at `m_flow × design flow × (valve − v0)/(100 − v0)`.
  Output = Ċ (T_flow − T_air)(1 − exp(−UA/Ċ)), with UA from the catalogue curve × `m_rad`.
- **floor heating**: `m_ufh × K_ws × area × (T_water − T_slab) × pump × circ`.
- **heat-source zone**: `UA_hp × (T_water − T_zone) × circ` (unit and pipe losses).
- **sun**: `g_sol × solar proxy`, spread over windows by area × orientation weight.
- **background gain**: `Q_int`, spread over occupied zones by floor area.
- **hot-water tank**: `cop_dhw × P_el` in, tap draw `tap_W` out; the source heat
  `(cop_dhw − 1) × P_el` is counted as drawn from the heat source.

A sensor on a radiator valve (`sensor_on_valve`) reads warm while its radiator emits:
- ntu model: air temperature plus `kappa_q` K per 100 W of radiator output, the sum passed through
  a first-order lag with time constant `valve_sensor_tau_h` (the sensor body);
- catalogue model: air temperature plus `kappa_s` × opening × (T_water − T_air), no lag.

## Parameters

Multipliers (`m_*`) scale a group of priors from the config: `m_env` all walls and roofs,
`m_vent` all ventilation, and so on. They, and the positive quantities `g_sol`, `eta_hp`,
`k_meter`, `gamma_v`, `cop_dhw` and `UA_tank`, are fitted on a log scale (`scale: log`). The rest
are fitted linearly (`T_ground`, `T_neighbour`, `Q_int`, `v0`, `kappa_q`, `kappa_s`, `P_aux_hp`,
`UA_hp`, `tap_W`, `dT_off`, `f_off_*`). The full list with meanings: `home-thermal check
house.yaml`, or [params.py](../packages/core/src/home_thermal/core/params.py). A parameter exists
only when something in the house uses it.

Fixed by default (held at the prior): `m_win`, `m_air`, `m_Cair`, `m_flow`, `dT_off` (prior 0:
set it to your measured meter offset), `cop_dhw`, `UA_tank`, `tap_W`. Room temperatures alone rarely separate these from other parameters; fitting
them lets them compensate for errors elsewhere. Free them per house under `parameters:` if you
have data that pins them down (and check on held-out data that it helps).

## Calibration

Maximum a posteriori (`home_thermal.calibrate.fit`): minimise

    Σ (hourly model − measured)² / σ²   +   Σ ((θ − θ_prior) / σ_prior)²

over the free parameters θ (log space for multipliers), within hard bounds. Data terms, all hourly
means after a warm-up while the initial state settles (`FitSettings.warmup`: 24 h in the Python
API, 48 h in the CLI and in `cross_validate`):

- every zone sensor, σ_T = 0.5 K,
- heat-meter output if `HP_heat_W` is present, σ = 800 W (meters are often biased; weak),
- heat-pump electricity if `HP_elec_W` and `T_flow` are present, σ = 100 W (reliable; strong).

Model errors are autocorrelated over hours, so the data term is down-weighted to about one
independent sample per 6 hours (`FitSettings.autocorr_h`). Otherwise the priors would be ignored.

The Laplace standard deviations from one fit are typically 2-5× too small. Use the spread across
cross-validation folds instead.

## Cross-validation

`cross_validate` cuts the data after the warm-up into blocks (default 3 days). For each block:
fit on everything else, simulate straight through, score the block with the fitted and with the
prior parameters. The report gives held-out RMSE per room and per fold, heat-pump errors, the
range of every parameter across folds, and whether it hit a bound. A change to the config or the
free parameters is an improvement only if the held-out RMSE goes down.

## Forecasting

`kalman` runs a Kalman filter on the network with one extra state per measured zone: unexplained
heat (W), a slow random walk standing in for everything not modelled. `forecast_errors` starts
the twin from the filtered state and scores three forecasts:

- `twin+filter`: the estimated unexplained heat held constant (best for the next hour),
- `twin+filter, d=0`: without it (usually best beyond a few hours: doors and people change),
- `persistence`: no change, the baseline to beat.

To forecast the real future, build a DataFrame with the expected inputs (outdoor temperature
forecast, planned setpoints or valve %, T_flow from your heating curve) for the next hours, and
run `simulate(net, future, x0=XS.iloc[-1].to_numpy())` with `XS, DS = kalman(net, recent_data)`.
For closed-loop control in the forecast, add `controller=Controller(...)`.

The filter noise levels (`FilterNoise`) are fixed engineering values, not tuned on the data.

## Controls

`home_thermal.control.Controller` replaces measured valve and pump columns step by step with what
the house's controls would do, given the twin's simulated sensor readings and the setpoint
columns. Thermostats are PI controllers with anti-windup; their tuning is fitted on recorded
valve % (open loop), separately from the thermal model, so it adds no freedom to the thermal fit.
Not modelled: the heat source's own flow-temperature control (`T_flow` comes from the data).
