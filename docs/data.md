# Input data

The model reads one table: a pandas DataFrame (or a CSV with the timestamp as first column) on a
**regular UTC time grid**. 15 minutes is a good step; 5-60 minutes all work. Each value is the
mean over the step that starts at the timestamp.

**How much:** at least a week of heating-season data for a fit; cross-validation needs the
warm-up (default 48 h) plus at least two blocks (default 3 days each), so about 8 days minimum,
and two to four weeks for useful parameter spreads. Cold, varied weather helps.

## Columns

Fixed names:

| column | unit | needed for | meaning |
|---|---|---|---|
| `T_out` | °C | always | outdoor temperature |
| `T_flow` | °C | radiators (ntu); heat-pump electricity term | water temperature leaving the heat source |
| `T_return` | °C | optional | water returning to it |
| `T_water` | °C | any emitter | mean water temperature while circulating; derived from T_flow/T_return if absent |
| `circ` | 0-1 | any emitter | fraction of the step that water circulated |
| `HP_heat_W` | W | optional calibration term | heat-meter output of the heat source |
| `HP_elec_W` | W | optional calibration term (with `T_flow`) | electricity used by the heat source |
| `HP_heat_per_K` | W/K | optional | HP_heat_W / raw (flow − return) ΔT, for the meter offset correction `dT_off` |

The two heat-pump terms are independent: with only `HP_heat_W` the fit uses the heat meter, with
only `HP_elec_W` + `T_flow` it uses electricity. Without either, it uses room temperatures alone
and does not fit the heat-pump parameters.

Names you choose in the house config:

| config key | content |
|---|---|
| `zones.<z>.sensor` | room temperature, °C |
| `emitters.radiators.<z>.valve` | valve opening, 0-100 % (or a fixed number in the config) |
| `emitters.floor_heating.<z>.pump` | pump on-fraction, 0-1 |
| `solar.proxy` (default `PV_W`) | sunshine proxy, e.g. PV output in W |
| `dhw.power` | hot-water heat pump electricity, W |
| `controls.thermostats.<z>.setpoint` | thermostat setpoint, °C (for control what-ifs) |
| `background_gain.meter_column` | household electricity, W (meter mode only) |

`home-thermal check house.yaml` prints the columns your house needs.

**No flow sensor?** `circ` can come from anything that says when the heating pump runs: a smart
plug on the pump, the boiler's OpenTherm "CH active" flag, or `T_flow − T_return > 1 K`.

Missing values (NaN) are allowed: in sensors the step is simply not scored (the start state uses
each sensor's first value within 6 hours), in inputs they count as off / zero. Not allowed: gaps
in the time grid itself. Resample first:

```python
df = raw.resample("15min").mean()
```

## From Home Assistant

`home_thermal.hass` reads the recorder database (`home-assistant_v2.db`, SQLite) directly,
read-only. A YAML file maps entities to columns:

```yaml
host: root@homeassistant.local     # optional, for `home-thermal hass-fetch` over ssh
ssh_key: ~/.ssh/id_homeassistant
entities:
  T_living:  sensor.living_temperature
  T_out:     sensor.outdoor_temperature
  V_bedroom: sensor.bedroom_heating_power
  T_flow:    sensor.heatpump_flow_temperature
  T_return:  sensor.heatpump_return_temperature
  water_flow: sensor.heatpump_water_flow
setpoints:
  SP_living: climate.living
flow_column: water_flow            # circ = flow above circulating_above (L/h)
circulating_above: 100
statistics_start: "2026-11-01"     # use hourly statistics before the ~10-day states window
```

All keys: `HassConfig` in [source.py](../packages/hass/src/home_thermal/hass/source.py); a
[worked example](../examples/semi_detached_1935/homeassistant.yaml).

- `on`/`off`/`heat`/`idle` states become 1/0; `unavailable` stays missing (unless an hourly
  statistic exists for that hour and `statistics_start` is set: then the statistic fills it).
- States are held until the next change (that is how the recorder stores them), sampled per
  minute and averaged per step.
- The recorder keeps full-resolution states ~10 days. Older data comes from the hourly long-term
  statistics (`statistics_start`), which are coarser: pump states become hourly on-fractions from
  plug power (`pump_running_W`). Hours missing from the statistics stay missing.
- Power statistics stored in kW are converted to W automatically. If an entity's *state* is in
  kW, list its column under `kw_columns`.
- `water_sensor_offset_K`: if the flow sensor reads higher than the return sensor with the pump
  stopped, put that difference here; both sensors are corrected by half. If your heat meter
  computes heat from the raw sensors, also set the same value as the prior of `dT_off` in the
  house config (`parameters: {dT_off: {prior: 0.85}}`).
- `pv_energy`: derive `PV_W` from an energy counter (kWh) instead of a power sensor.

```bash
home-thermal hass-fetch homeassistant.yaml --dest data/ha_db       # copy the DB over ssh
home-thermal hass-load homeassistant.yaml --db data/ha_db/home-assistant_v2.db --out data/data.csv
```

`hass-fetch` copies the live database files with `tar`; if Home Assistant writes at that moment
the copy can be slightly inconsistent. Run it again if loading fails.

## From anything else

Write the columns above to a CSV and use `home_thermal.core.read_csv`, or pass a DataFrame to
the Python API. Nothing else in the library depends on where the data comes from.

## Synthetic data

`home_thermal.core.synthetic.synthetic_data(cfg, true_params, days=10, solar=True, heat_pump=True)`
generates data for any house config from known parameters, with simple thermostats. Use it to
try a config before you have data, or to see which parameters your measurements can identify.
