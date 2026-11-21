"""Heat-pump performance: COP as a fixed fraction of the Carnot COP.

    COP = eta * T_cond / (T_cond - T_evap),  T_cond = T_flow + dT_cond,  T_evap = T_out - dT_evap
    electric power while running = P_aux + Q / COP;  standby otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HeatPump:
    eta: float = 0.5           # fraction of Carnot
    P_aux: float = 50.0        # W, pump/fan/controls while running
    dT_cond: float = 3.0
    dT_evap: float = 6.0
    standby_W: float = 7.0
    running_min_W: float = 50.0   # heat output above which the unit counts as running

    @classmethod
    def from_config(cls, cfg: dict, p: dict) -> HeatPump:
        hp = cfg["heat_pump"]
        return cls(eta=p["eta_hp"], P_aux=p["P_aux_hp"], dT_cond=hp["dT_cond"], dT_evap=hp["dT_evap"],
                   standby_W=hp["standby_W"], running_min_W=hp["running_min_W"])

    def carnot(self, T_out, T_flow):
        tc = np.asarray(T_flow) + self.dT_cond + 273.15
        te = np.asarray(T_out) - self.dT_evap + 273.15
        return tc / np.maximum(tc - te, 3.0)

    def cop(self, T_out, T_flow):
        return self.eta * self.carnot(T_out, T_flow)

    def electric(self, Q, T_out, T_flow):
        """Electric power (W) for heat output Q (W) at the given temperatures."""
        Q = np.asarray(Q, float)
        return np.where(Q > self.running_min_W, self.P_aux + Q / self.cop(T_out, T_flow), self.standby_W)

    def electric_hourly(self, Q: pd.Series, df: pd.DataFrame) -> pd.Series:
        """Hourly electricity implied by a heat series Q, using df's T_out and T_flow."""
        h = pd.DataFrame({"Q": Q, "To": df["T_out"], "Tf": df["T_flow"]}).resample("1h").mean()
        return pd.Series(self.electric(h.Q, h.To, h.Tf), index=h.index)
