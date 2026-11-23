# Hydraulics: modelling the heating circuit

`home_thermal.hydraulics` (package `home-thermal-hydraulics`) models the water side of a heating
system: which way and how much water flows, and at what temperature it reaches each emitter and
returns to the heat source. It is independent of the thermal model; the two meet through the
step inputs (`ctx`): room and slab temperatures in, flows and return temperatures out.

## Components

All flows in m³/h, pressures in kPa (1 m head ≈ 9.78 kPa, 0.4 bar = 40 kPa), kv in m³/h at 1 bar.

| component | what it models | key parameters |
|---|---|---|
| `Pump` | circulator, modes `speed` (constant speed, setting = speed fraction), `dp-c` (constant head, setting in m), `dp-v` (head falls to half at zero flow); a stopped pump is a resistance `kv_off` | `PumpCurve(H0, q0)`: shut-off head (m), flow at zero head (m³/h) |
| `Kv` | fixed resistance (fittings, heat exchanger, a lumped pipe run) | `kv` |
| `Pipe` | Darcy-Weisbach pipe plus local losses | length, inner diameter, `zeta` |
| `Valve` | control valve, opening from a number or a data column (with a dead band) | `kv_max`, characteristic linear / equal / quick |
| `OverflowValve` | differential-pressure bypass: closed below `p_open`, opens over a band | `p_open` (kPa, may come from a column), `kv_max` |
| `ThermostaticMixingValve` | wax-element 3-way mixing valve on a manifold, proportional band | `t_set`, `kv`, `band` |
| `HeatSource` | heats to a set outlet temperature and/or by a power (negative: takes heat out) | `t_out`, `power` |
| `Emitter` | radiator (q50, n) or floor loop / heat exchanger (UA) giving heat to a space | `space`, `UA` or `q50` + `n` |
| `Buffer` | fully mixed tank in series; advanced by `Circuit.step` | `volume_l`, optional `t_column` (measured temperature) |

`PUMP_CURVES` holds approximate maximum curves for common 6 m and 4 m wet-rotor circulators
(e.g. Wilo Para 25/6). Replace them with your pump's datasheet values when you have them.

## Composing a circuit

In Python:

```python
from home_thermal.hydraulics import Circuit, Pump, PumpCurve, HeatSource, Kv, Valve, Emitter, OverflowValve

c = Circuit(reference="hp_in")                     # node where the expansion vessel connects
c.add("hp", "hp_in", "hp_out", [Pump(PumpCurve(7, 2.5)), HeatSource(t_out=35)])
c.add("rad", "hp_out", "hp_in", [Valve(0.8, opening=0.6), Emitter("T_bedroom", q50=1500)])
c.add_overflow("bypass", "hp_out", "hp_in", OverflowValve(p_open=40, kv_max=1.5))
res = c.solve({"T_bedroom": 19.0})
res.flow["bypass"], res.temperature["hp_in"], res.heat
```

Or declared in YAML ([full example](../examples/semi_detached_1935/hydraulics.yaml)) and built
with `build_circuit(load_spec("hydraulics.yaml"))`. In YAML:

- a **branch** is a list of elements in series between two nodes;
- `overflow:` makes a branch an overflow valve;
- `mixing_group:` expands into a manifold with a thermostatic mixing valve, its pump, one or more
  loop groups and the return to the primary circuit;
- values can be numbers, `$parameter` references (calibratable, with `value` and `bounds` under
  `parameters:`), or per-step inputs `{column: NAME, scale:, offset:, above:, dead_band:}`.

## Solving

`solve(ctx)`:
1. pressures and flows: Newton (MINPACK hybrid) on branch pressure-drop equations and node mass
   balances, with an analytic Jacobian, warm-started from the previous step;
2. temperatures: advection with mixing at nodes, iterated until steady;
3. thermostatic mixing valves: each settles where its opening matches what its wax element wants
   at the resulting mixed temperature (a monotone equation, solved by bracketing).

`step(ctx, dt)` also advances buffers. A typical house circuit solves in 50-100 ms.

## Calibration

`home_thermal.hydraulics.calibrate.fit_circuit(spec, ctxs, observed, predict)` fits the
`$parameters` to any measured quantity (e.g. the heat pump's water flow), within bounds and with a
weak pull toward the starting values. Check the result on held-out rows against a simple baseline
(e.g. the mean flow per pump state).

**A flow sensor alone cannot fix the circuit's absolute scale**: a sensor reading 20 % low makes
all kv values and pump flows 20 % low with an equally good fit. Give the fit one absolute
reference: a pump datasheet curve, the flow indicators on a floor-heating manifold (L/min per
loop) read in a known state, or a heat meter with its own flow measurement.
