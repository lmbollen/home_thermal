# House configuration (YAML)

One file describes the house. Only `zones` is required; everything else has a default. Check it
with `home-thermal check house.yaml` (or `home_thermal.core.load_house`) after every edit: errors
name the key that is wrong, unknown keys are rejected (with a "did you mean" hint), and the output
lists the parameters and data columns your house needs. Any mapping may carry a free-text `note`.

Units: m, m², m³, W, W/K, W/m²K, °C.

Every number is a **prior**, your best estimate. Calibration scales groups of them with
multipliers (see [model.md](model.md)), so rough values are fine as long as their relative sizes
are sensible. [Where the numbers come from](#where-the-numbers-come-from) at the end helps.

```yaml
name: my house
zones: { ... }          # required
envelope: { ... }
internal: [ ... ]
emitters: { ... }
heat_pump: { ... }      # or boiler: { ... }
dhw: { ... }
solar: { ... }
background_gain: { ... }
boundaries: { ... }
thermal_mass: { ... }
parameters: { ... }
controls: { ... }
```

## zones

A zone is a room or a group of rooms treated as one temperature. Rules of thumb:

- one zone per temperature sensor;
- plus a zone for each important unmeasured space that sits between measured ones (a hall and
  staircase, a utility room), so heat can flow through it;
- merge small rooms that are always open to each other; split rooms that are heated differently.

```yaml
zones:
  living:  {area: 30, volume: 75, ach: 0.6, sensor: T_living}
  hall:    {area: 8, volume: 30, ach: 1.0, occupied: false}
  shed:    {area: 14, volume: 35, sensor: T_shed, occupied: false, in_house_loss: false}
```

| key | default | meaning |
|---|---|---|
| `area` | required | floor area, m² (scales thermal mass and the background-gain share) |
| `volume` | required | air volume, m³ |
| `ach` | 0.5 | air changes per hour to outside (infiltration + ventilation, after heat recovery) |
| `sensor` | none | data column with this zone's measured temperature |
| `occupied` | true | receives a share of the background gain (people, appliances) |
| `in_house_loss` | true | counted in `H_house` (set false for an attached shed or garage) |

At least one zone must have a sensor and at least one must be occupied.

## envelope

Per zone, a list of elements to the outside, the ground or a neighbour:

```yaml
envelope:
  living:
    - {kind: wall,   area: 20, U: 0.5, note: "cavity wall, insulated 1995"}
    - {kind: window, area: 5,  U: 1.6, solar: S}
    - {kind: floor,  area: 30, U: 0.3, ufh: true}
    - {kind: party,  area: 15, U: 1.2}
```

| kind | goes to | scaled by |
|---|---|---|
| `wall`, `roof` | outdoor air | `m_env` |
| `window`, `door` | outdoor air | `m_win` |
| `floor` | ground / crawl space (`T_ground`) | `m_floor` |
| `party` | a neighbour (`T_neighbour`) | `m_party` |

- `U` in W/m²K, including surface resistances. From an Rc value (energy labels give these):
  walls U ≈ 1/(Rc + 0.17), roofs ≈ 1/(Rc + 0.14), floors over a crawl space ≈ 1/(Rc + 0.34).
- `solar` (windows and doors only): facade orientation, one of S, SW, SE, W, E, N, NW, NE, H
  (horizontal / roof window). Add your own under `solar.weights`.
- `ufh: true` on the floor element of a floor-heated zone connects the ground loss to the heated
  slab instead of the room air.

**Apartments.** Floors and ceilings to other heated flats, and walls to neighbours, are `party`
elements; they all share one temperature `T_neighbour` (fitted). Set its prior near the
neighbours' temperature: `boundaries: {T_neighbour: 19}`. An unheated stairwell or a garage
below is better as its own unmeasured zone (`occupied: false`, `in_house_loss: false`) with its
own envelope. In `flows`, a negative loss to "neighbour" means heat gained from the neighbours.

## internal

Heat paths between zones, W/K: `[zone_a, zone_b, UA]`, optionally with `air` (door or staircase
air exchange, scaled by `m_air`) or `solid` (walls and floors, scaled by `m_int`; the default).

```yaml
internal:
  - [living, hall, 100, air]      # door usually open
  - [hall, bedroom, 10, air]      # door usually closed
  - [living, bedroom, 20]         # timber floor/ceiling between them: ~1.5 W/m²K x area
```

- `solid`: U of the internal wall or floor × area. Brick inner wall ~2 W/m²K, timber floor
  ~1.5 W/m²K, stud wall ~1 W/m²K.
- `air`: an open doorway exchanges a lot of air by buoyancy, roughly 100 W/K for a normal door at
  1 K difference (more at larger differences); a closed door 5-10 W/K; an open staircase 50-150 W/K.
  `m_air` is fixed by default because door positions are not measured, so pick a typical value.

## emitters

```yaml
emitters:
  radiator_model: ntu            # or catalogue
  valve_sensor_tau_h: 0.5
  radiators:
    bedroom: {q50: 1200, n: 1.3, valve: V_bedroom, sensor_on_valve: true}
    kitchen: {q50: 900, valve: 100}        # no valve data: always fully open
  floor_heating:
    living: {area: 30, K_ws: 15, h_sa: 10.8, c: 130000, pump: P_living}
```

**Radiators** (one per zone; combine several radiators in a zone by adding their q50):

| key | default | meaning |
|---|---|---|
| `q50` | required | output at ΔT50 (75/65/20 °C), W; from the datasheet or the table below |
| `n` | 1.3 | radiator exponent: panel radiators 1.25-1.35, convectors ~1.4, fan-assisted ~1.0-1.1 |
| `valve` | required | data column with the valve opening 0-100 %, or a fixed number (e.g. `100` for a radiator with an ordinary thermostatic valve and no data; it then acts as always open) |
| `sensor_on_valve` | false | the zone's sensor sits on this radiator's valve (smart radiator thermostat): it reads warm while the radiator emits |
| `flow_lph` | q50 × 3600 / (4186 × 10) | water flow at 100 % in L/h, i.e. the flow for a 10 K drop at q50 (ntu model) |

`radiator_model`:
- `ntu` (default): water enters at `T_flow`; flow is proportional to the valve opening above a
  dead band `v0`. Output is flow-limited at small openings, surface-limited at large ones. Use
  this with smart thermostats that report a valve %.
- `catalogue`: output = q50 (ΔT/49.83)^n (valve/100)^γ at the mean water temperature.

**Floor heating** (one circuit per zone), all four numbers required:

| key | typical | meaning |
|---|---|---|
| `area` | | heated floor area, m² |
| `K_ws` | 15 (15 cm pipe spacing) - 25 (10 cm) | water → slab conductance, W/m²K |
| `h_sa` | 10.8 bare screed/tiles; ~7 laminate; ~5 parquet or carpet | slab surface → room, W/m²K (put the floor covering here) |
| `c` | 130 000 (6-7 cm screed) | slab heat capacity, J/m²K |
| `pump` | `always` (default) | data column with the circuit pump's on-fraction 0-1, or `always` |

Each pump column gets a parameter `f_off_<pump>`: the residual flow while the pump is off.

## heat_pump

```yaml
heat_pump: {zone: shed, dT_cond: 3, dT_evap: 6, standby_W: 7, running_min_W: 50}
```

Used only when the data has heat-pump measurements (`HP_heat_W` and/or `HP_elec_W`): COP =
η × Carnot(T_flow + dT_cond, T_out − dT_evap); below `running_min_W` of heat the unit counts as
off and uses `standby_W`. `zone` (optional): where the indoor unit and pipework lose heat
(parameter `UA_hp`).

## boiler

A gas boiler instead of a heat pump (configure one of the two):

```yaml
boiler:
  gas_column: gas_m3        # default; gas used per step
  gas_unit: m3              # m3 or kWh per step, or W (power)
  kwh_per_m3: 9.77          # upper calorific value; Dutch natural gas 9.77, UK/EU typically 10.5-11.2
  dhw_flag: dhw_active      # optional: column that is 1 while a combi boiler heats tap water
  zone: utility             # optional: where the boiler and its pipes lose heat (parameter UA_boiler)
parameters:
  eta_boiler: {prior: 0.9}  # efficiency on the upper calorific value
```

Calibration then compares hourly gas use with the modelled heat ÷ `eta_boiler`. The gas meter is
a strong extra constraint on the house's heat demand, and so on `H_house` (in tests it brought a
deliberately wrong prior to within ~5 % of the truth; heat can still hide in the background gain
and the ground/neighbour paths).

**Tap water.** Gas used for tap water must be kept out: whole hours are left out when the
`dhw_flag` column is set in any step (OpenTherm "DHW active", a three-way valve position, a
cylinder thermostat), and when gas burned while the heating did not run at all that hour. Short
draws inside heating hours are only found with the flag, so configure one if you can, for combi
boilers and for system boilers that reheat a hot-water cylinder alike. (The `dhw:` section is for
a separate heat-pump water heater, not for a boiler's cylinder.)

**Efficiency.** `eta_boiler` is **fixed** by default. Room temperatures and gas mostly determine
heat ÷ efficiency, and efficiency trades off against the radiator output (`m_rad`), so set its
prior from the boiler type: modern condensing boiler at low flow temperatures 0.90-0.95,
condensing boiler with high flow temperatures (> 55 °C return) 0.85, old non-condensing
0.75-0.80 (all on the upper calorific value, as gas bills use). You can free it
(`--free eta_boiler`); check on held-out data that it helps.

You need `T_flow` (or `T_water`) and `circ` (1 while the boiler's heating pump runs, e.g. from
OpenTherm or a smart plug on the pump). See the
[apartment example](../examples/apartment_gas_boiler/house.yaml).

## dhw (optional)

A hot-water tank heated by a water/water heat pump that takes its source heat from the heating
circuit (so that heat shows up in the heat meter):

```yaml
dhw: {zone: shed, volume_l: 200, power: DHW_W, on_threshold_W: 5, T_cold: 10, T_max: 60}
```

`power`: data column with its electricity use (W). Leave the section out if hot water is separate.

## solar

```yaml
solar: {proxy: PV_W, weights: {S: 1.0, N: 0.2}}
```

`proxy`: a data column proportional to sunshine. PV output works well; a pyranometer or a
weather-service irradiance column works too. Without a proxy column there is no solar gain (sunny
days then look like extra background gain; expect larger errors). `g_sol` (W of solar gain per
unit of proxy) is fitted. A good prior with PV as the proxy:

    g_sol ≈ 0.4 × Σ(window area × orientation weight) [m²] / PV peak power [kWp]

(e.g. 10 m² weighted glass and 0.63 kWp gives 6, the default). Set it under `parameters`.

## background_gain

```yaml
background_gain: {mode: constant}
background_gain: {mode: meter, meter_column: P_app_W, meter_heat_share: 0.8}
```

`constant`: one fitted average (`Q_int`, W). `meter`: plus a fixed share of a household-electricity
column. Neither models people or appliances in detail, on purpose (see the README).

## boundaries

```yaml
boundaries: {T_ground: 10, T_neighbour: 19}
```

Priors for the ground and neighbour temperatures (defaults 11 and 16 °C; fitted within ±3 and
±6 K of them). Both are fitted unless
fixed under `parameters`. They exist only if the house has floor / party elements.

## thermal_mass

```yaml
thermal_mass: {air_furniture_factor: 5, C_m_per_m2: 165000, H_am_per_m2: 22.75}
```

ISO 13790 classes per m² floor area: light C_m 110 000 / H_am 22.75 (timber frame), medium
165 000 / 22.75 (default), heavy 260 000 / 27.3 (concrete floors and walls).

## parameters

Override any parameter's prior, uncertainty, bounds, or whether it is fitted:

```yaml
parameters:
  g_sol:  {prior: 3.0}                # smaller PV array
  m_win:  {fixed: false}              # fit windows too
  T_neighbour: {prior: 19, sigma: 1, bounds: [15, 23]}
  dT_off: {prior: 0.85}               # measured heat-meter sensor offset
```

Fields: `prior`, `sigma`, `bounds` (`[low, high]` or `null` for none), `fixed`. For multipliers
and the other log-scale parameters, `sigma` is in log space (0.25 ≈ ±25 %); for linear ones it is
in the parameter's unit. `home-thermal check` lists every parameter of your house with its meaning,
and [params.py](../packages/core/src/home_thermal/core/params.py) shows each one's scale.

## controls (for closed-loop what-ifs)

```yaml
controls:
  thermostats:                         # PI controllers that write valve columns
    living:  {valve: V_living, setpoint: SP_living, group: downstairs}
    bedroom: {valve: V_bedroom, setpoint: SP_bedroom, group: radiators}
  hysteresis:                          # on/off thermostats that switch a pump column
    P_shed: {zone: shed, setpoint: SP_shed, below: 0.3, above: 0.3}
  pumps:                               # observed (data) | always | follow (on while a valve > 0)
    P_living: {mode: follow, follow: V_living}
```

A thermostat's `valve` is the column it writes: a radiator's valve, or a column a pump follows
(as for the living room above, where the floor pump follows the thermostat). Thermostats in the
same `group` share one tuning (Kp, Ti), fitted with `home_thermal.control.fit_pi` on recorded
valve %. Defaults: `valve: V_<zone>`, `setpoint: SP_<zone>`, `group: <zone>`.

## Where the numbers come from

| quantity | source / rule of thumb |
|---|---|
| areas, volumes | floor plans; volume = area × ceiling height |
| U-values | energy label (Rc → U above), building-era tables, or: uninsulated cavity wall 1.5, insulated cavity 0.5, single glazing 5.1, double 2.9, HR++ 1.2-1.6, triple 0.8 |
| `ach` | blower-door test: ach ≈ n50 / 20. Otherwise: pre-1970 house 0.8-1.2, 1970-2000 0.5-0.8, airtight new build with mechanical ventilation 0.3-0.5, with heat recovery (MVHR) 0.1-0.2 (the share not recovered) |
| radiator `q50` | datasheet; or per metre length at ΔT50, 600 mm high: type 11 ≈ 900 W, type 21 ≈ 1300 W, type 22 ≈ 1650 W, type 33 ≈ 2400 W (scale roughly with height) |
| floor heating | see the table above |
| internal UA | see [internal](#internal) |
| `g_sol` | formula under [solar](#solar) |
