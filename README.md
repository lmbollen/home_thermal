# home_thermal

A thermal digital twin of your house: describe it in one YAML file, feed it measured data (room
temperatures, outdoor temperature, heating), and get a model that

- predicts room temperatures hours to a day ahead,
- tells you how much heat leaves through walls, windows, ventilation, floor and party walls,
- shows how rooms heat each other, and
- simulates what-ifs: other setpoints, another pump strategy, another thermostat tuning.

It is a *grey-box* model: a resistor-capacitor network built from what you know about the house
(areas, U-values, radiator sizes), with a small set of multipliers fitted to your measurements.
Every result comes with an honest held-out accuracy (blocked cross-validation), so you can tell
what the data supports and what it does not.

## Packages

One repository, six installable packages under the `home_thermal` namespace. Install only what
you need.

| Package (pip name) | Import | What it does | Depends on |
|---|---|---|---|
| `home-thermal-core` | `home_thermal.core` | House config, RC network, heat pump / gas boiler, simulation, energy balance | numpy, scipy, pandas, pyyaml |
| `home-thermal-calibrate` | `home_thermal.calibrate` | Fit to data, cross-validation, Kalman filter, forecasts | core |
| `home-thermal-control` | `home_thermal.control` | Thermostats (PI, hysteresis) and pump logic for closed-loop what-ifs | core |
| `home-thermal-hydraulics` | `home_thermal.hydraulics` | Heating circuit: pumps, valves, mixing valves, bypass, buffer; flows and water temperatures ([docs](docs/hydraulics.md)) | numpy, scipy, pyyaml |
| `home-thermal-hass` | `home_thermal.hass` | Data from a Home Assistant recorder database | pandas, pyyaml |
| `home-thermal` | `home_thermal.cli` | The `home-thermal` command, uses all of the above | all |

The packages are not on PyPI yet; install from a clone:

```bash
git clone https://github.com/lmbollen/home_thermal
cd home_thermal
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e packages/core -e packages/calibrate -e packages/control -e packages/hass -e packages/hydraulics -e packages/cli
```

Python 3.11 or newer.

## Five-minute tour (no data needed)

The demo writes a small made-up house and twelve days of synthetic measurements:

```bash
home-thermal demo demo/                    # demo/house.yaml, demo/data.csv, demo/truth.json
home-thermal check demo/house.yaml         # what the model looks like, which columns it needs
home-thermal crossval demo/house.yaml demo/data.csv --out-dir demo/results
home-thermal flows demo/house.yaml demo/data.csv --params demo/results/params.json
```

`crossval` (about a minute) prints the held-out accuracy per room, the parameter spread across
folds, and a final fit on all data, and writes `report.txt`, `params.json` and CSVs to
`demo/results`. `flows` shows where the heat came from and went.

The demo data was generated from known parameters (`truth.json`). Compare: the fitted `H_house`
(total heat-loss coefficient) lands close to the true one, while individual multipliers such as
`m_env` and `m_vent` do not. That is the typical picture with real data too: the total is well
determined, the split is not (see [How far to trust it](#how-far-to-trust-it)).

The same in Python:

```python
from home_thermal.core import Network, load_house, read_csv, simulate, sensor_temps
from home_thermal.calibrate import fit, cross_validate

cfg = load_house("demo/house.yaml")
df = read_csv("demo/data.csv")

result = fit(cfg, df)                       # MAP fit of the default free parameters
net = Network(cfg, result.params)
print(f"heat loss coefficient {net.H_house():.0f} W/K")

states, heat = simulate(net, df)            # node temperatures and heat delivered
predicted = sensor_temps(net, states, df)   # what each room sensor should read

cv = cross_validate(cfg, df)                # is it any good on days it has not seen?
print(cv.report(cfg))
```

## Modelling your own house

1. **Describe the house** in a YAML file: zones (rooms or groups of rooms), envelope elements per
   zone, links between zones, emitters (radiators, floor heating). Start from
   [the demo house](packages/core/src/home_thermal/core/examples/demo_house.yaml) or the
   fully worked examples: a [semi-detached house with a heat pump](examples/semi_detached_1935/house.yaml)
   and an [apartment with a gas combi boiler](examples/apartment_gas_boiler/house.yaml). Every key is described in
   [docs/configuration.md](docs/configuration.md). Run `home-thermal check house.yaml` after every
   edit: it explains what is wrong and lists the data columns the model needs.
2. **Get the data** into one table on a regular time grid (15 minutes works well), with the column
   names your config uses: see [docs/data.md](docs/data.md). With Home Assistant, write a small
   mapping file (entity -> column, [example](examples/semi_detached_1935/homeassistant.yaml)) and run
   `home-thermal hass-load homeassistant.yaml --db home-assistant_v2.db --out data.csv`.
   Anything else (a CSV export, another logger) works as long as the columns match.
   At least a week of heating-season data; two or more is better.
3. **Cross-validate and fit**: `home-thermal crossval house.yaml data.csv` (or just
   `home-thermal fit house.yaml data.csv --out params.json`). `--fix a,b` holds parameters at their
   prior, `--free c` fits an extra one. Read the report:
   - *held-out room RMSE*: the fitted model should beat the priors in most folds. With decent
     sensors, 0.3-0.6 K per room is good, around 1 K is usable, much more means something big is
     missing from the config (a heat source, a wrong emitter, a sensor in the sun);
   - *forecasts*: the twin should beat "no change" (persistence) at 6 and 24 hours;
   - *parameters*: values that hit a bound (`at_bound`) or vary a lot across folds are not
     identified by your data. Fix them at a sensible prior (`--fix`, or `fixed: true` under
     `parameters:`), rather than widening the bounds, unless you know the prior was wrong.
4. **Use it**: `simulate`, `flows`, forecasts and closed-loop what-ifs (below).

### A what-if in Python

```python
from home_thermal.calibrate import load_params
from home_thermal.control import Controller, fit_pi
from home_thermal.core import Network, load_house, read_csv, sensor_temps, simulate

cfg = load_house("house.yaml")
df = read_csv("data.csv")
net = Network(cfg, load_params("results/params.json", cfg))

tuning = fit_pi(cfg, df)                    # thermostat Kp/Ti from recorded valve %
scenario = df.copy()
scenario["SP_living"] += 1.0                # one degree warmer in the living room
ctl = Controller(net, scenario, tuning)
states, heat = simulate(net, scenario, controller=ctl)
rooms = sensor_temps(net, states, ctl.inputs())
print(f"mean heat {heat.mean():.0f} W")
```

## How far to trust it

A house has many heat flows nobody measures: open doors and windows, people, cooking, a laptop.
This library deliberately does **not** model them. One constant background gain keeps their
average from being credited to the insulation, and the inaccuracy they cause is accepted. More
parameters would fit the past better and predict the future worse.

So: a parameter is fitted only if the data can pin it down; the rest stay at their prior
(`fixed` in [params.py](packages/core/src/home_thermal/core/params.py), changeable per house).
Judge changes on held-out data, never on the fit itself. Typical findings:

- The total heat-loss coefficient (`H_house`) is well determined; how it splits between walls,
  ventilation and floor often is not. Quote totals, not the split, unless the report shows the
  individual multipliers are stable across folds.
- Parameters that hit an optimiser bound in some fold (`at_bound` in the report) are compensating
  for something the model does not know. Fix them, or add a measurement that separates them.
- Absolute heat-pump efficiency and heat-meter scale trade off against each other.

See [docs/model.md](docs/model.md) for the physics and the calibration method.

## Development

```bash
pip install pytest ruff
pytest           # ~30 s, synthetic data only
ruff check packages tests
```

Tests never need real data: they generate a synthetic house with known parameters
(`home_thermal.core.synthetic`) and check that the library recovers it.

## Licence

MIT, see [LICENSE](LICENSE).
