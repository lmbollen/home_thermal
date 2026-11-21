"""Gas boiler: gas power = heat to the water / efficiency (constant efficiency, on the upper
calorific value, which is what gas bills and meters use).

Gas data comes as an amount per step (m3 or kWh) or as power (W); `gas_power` converts it to W.
Boilers also heat tap water with the same gas (combi boilers directly, system boilers by reheating a
cylinder); `dhw_hours` marks hours with tap-water heating so that calibration can leave them out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BURNING_W = 100.0   # hourly mean gas power above which the burner counted as on (~0.01 m3/h)


@dataclass
class Boiler:
    eta: float = 0.9

    @classmethod
    def from_config(cls, cfg: dict, p: dict) -> Boiler:
        return cls(eta=p["eta_boiler"])

    def fuel(self, Q):
        """Gas power (W) for heat to the water Q (W)."""
        return np.asarray(Q, float) / self.eta


def gas_power(cfg: dict, df: pd.DataFrame) -> pd.Series:
    """The boiler's gas use as power (W), from the configured gas column and unit."""
    b = cfg["boiler"]
    g = df[b["gas_column"]].astype(float)
    if b["gas_unit"] == "W":
        return g
    step_h = (df.index[1] - df.index[0]).total_seconds() / 3600
    kwh = g * b["kwh_per_m3"] if b["gas_unit"] == "m3" else g
    return kwh * 1000.0 / step_h


def dhw_hours(cfg: dict, df: pd.DataFrame) -> pd.Series:
    """Hours (hourly index) in which the boiler heated tap water:
    - any step with the `dhw_flag` column set (the reliable way: OpenTherm 'DHW active', a
      three-way valve position, a cylinder thermostat), and
    - hours in which gas burned while the heating did not run at all (circ 0 all hour).
    Short draws inside heating hours are only found with the flag."""
    b = cfg["boiler"]
    hourly = df.resample("1h")
    flag = (df[b["dhw_flag"]].fillna(0) > 0).resample("1h").max() if b["dhw_flag"] in df else False
    burning = gas_power(cfg, df).fillna(0).resample("1h").mean() > BURNING_W
    no_heating = hourly["circ"].max().fillna(0) <= 0 if "circ" in df else False
    out = burning & no_heating
    return (out | flag).astype(bool) if isinstance(flag, pd.Series) else out.astype(bool)
