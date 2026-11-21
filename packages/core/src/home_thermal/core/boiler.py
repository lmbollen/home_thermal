"""Gas boiler: gas power = heat to the water / efficiency (constant efficiency, on the upper
calorific value, which is what gas bills and meters use).

Gas data comes as an amount per step (m3 or kWh) or as power (W); `gas_power` converts it to W.
Combination boilers also heat tap water with the same gas; `dhw_steps` marks those steps so that
calibration can leave them out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BURNING_W = 50.0    # gas power above which the burner counts as on


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


def dhw_steps(cfg: dict, df: pd.DataFrame) -> pd.Series:
    """Steps where the boiler (probably) heated tap water: the dhw_flag column if configured,
    plus steps with gas use while no heating water circulated."""
    b = cfg["boiler"]
    if b["dhw_flag"] and b["dhw_flag"] in df:
        flag = df[b["dhw_flag"]].fillna(0) > 0
    else:
        flag = pd.Series(False, index=df.index)
    burning = gas_power(cfg, df).fillna(0) > BURNING_W
    no_heating = df["circ"].fillna(0) <= 0 if "circ" in df else pd.Series(False, index=df.index)
    return flag | (burning & no_heating)
