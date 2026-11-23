# Example: semi-detached house, 1935

The house this library was developed on: a 1935 semi-detached house, renovated in the 1990s.

- 9 zones: living/kitchen, hall + stairs (unheated, unmeasured), utility room, a heated shed with
  the heat pump's indoor parts, three bedrooms/rooms, bathroom, attic.
- Air-water heat pump (7 kW R290 monoblock) with a heat meter and power reading.
- Floor heating downstairs on two pumps (living room; utility room + shed), always-on bathroom
  floor loop.
- Radiators upstairs with smart radiator thermostats; their sensors sit on the valve
  (`sensor_on_valve: true`), so they read warm while the radiator emits.
- A 200 L water/water hot-water heat pump that draws its source heat from the heating return.
- PV output as the solar proxy.

Files:

- [house.yaml](house.yaml): the house. Every value is a prior from the energy label, the
  renovation drawing and the owner's notes; comments say where a number came from.
- [homeassistant.yaml](homeassistant.yaml): how its Home Assistant entities map to data columns.

Measured data and fitted results are not in the repository. With access to the Home Assistant
instance:

```bash
home-thermal hass-fetch homeassistant.yaml --dest data/ha_db
home-thermal hass-load homeassistant.yaml --db data/ha_db/home-assistant_v2.db --out data/data.csv
home-thermal crossval house.yaml data/data.csv --out-dir results
```

Things this house taught the model (now generic features):

- valve-mounted sensors need their own sensor model, or upstairs rooms look 1-2 K too warm
  whenever the radiators run;
- radiator valves have a dead band (here about 23 %), below which no water flows;
- the heat meter's flow and return sensors differed by 0.85 K at standstill; correcting it
  (`water_sensor_offset_K` in homeassistant.yaml, `dT_off` prior in house.yaml) improved the
  electricity prediction;
- the heat pump's heat figure uses a fixed internal flow (~840 L/h), not the measured water flow;
  once the sensor offset is removed it gives a plausible COP (3.4-3.7 at 2 °C outside), so the
  meter scale `k_meter` is fixed at 1. A COP from the measured flow came out implausibly low
  (~2.1): treat the water-flow sensor with caution. When the integration started delivering
  corrected values, `water_offset_until` stopped the loader from correcting them twice;
- an attached shed with the heat pump in it needs `in_house_loss: false`, or its losses inflate
  the house's heat-loss coefficient.
