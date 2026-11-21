# Example: apartment with a gas combi boiler

A made-up middle-floor apartment from the 1970s, to show:

- **apartment boundaries**: floor, ceiling and walls to other flats are `party` elements, all at
  one fitted neighbour temperature with a prior of 19 °C (`boundaries:`);
- **a gas boiler** (`boiler:`): calibration uses the gas meter. Gas burned for tap water is left
  out: hours with the `dhw_active` flag, and hours where gas burned while the heating did not run;
- **a radiator without valve data** (`valve: 100`, the kitchen);
- **no solar proxy** (no PV): there is no solar gain in the model.

Try it without data:

```python
from home_thermal.core import load_house, Network
from home_thermal.core.synthetic import synthetic_data
from home_thermal.calibrate import fit

cfg = load_house("examples/apartment_gas_boiler/house.yaml")
truth = {"m_env": 1.3, "m_vent": 1.3}          # envelope and ventilation 30 % worse than the priors
df = synthetic_data(cfg, truth, days=10, solar=False)
res = fit(cfg, df)
print(Network(cfg).H_house(), Network(cfg, res.params).H_house(), Network(cfg, truth).H_house())
```

Or from the command line: `home-thermal synth examples/apartment_gas_boiler/house.yaml data.csv
--no-solar --truth '{"m_env": 1.3}'`, then `home-thermal crossval ... data.csv`.

With Home Assistant: map the gas meter's total with `gas_counter` in the Home Assistant config,
which produces `gas_m3` per step (see [docs/data.md](../../docs/data.md)).
