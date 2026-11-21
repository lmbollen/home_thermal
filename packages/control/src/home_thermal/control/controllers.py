"""Controllers for what-if simulations: they replace measured valve % and pump states with what
the house's controls would do given the twin's own simulated sensor readings and setpoints.

Configured under `controls:` in the house YAML:

    thermostats:   PI controllers per zone (smart radiator / floor thermostats). Each writes a
                   valve % column. Zones in the same `group` share one (Kp, Ti), fitted with
                   `fit_pi` on recorded valve % (open loop, separate from the thermal fit).
    hysteresis:    on/off thermostats that switch a pump column (e.g. a generic thermostat).
    pumps:         pump columns: 'observed' (data), 'always' (1), or 'follow' (on while a valve
                   column is > 0).

Not modelled: the heat source's own flow-temperature control (T_flow comes from the data).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from home_thermal.core import Network, sensor_lag, step_seconds, valve_sensor_warming

PUMP_MODES = ("observed", "always", "follow")


@dataclass
class PI:
    """PI controller with output 0..100 % and conditional-integration anti-windup."""
    Kp: float = 100.0      # % per K
    Ti_h: float = 4.0      # integral time, hours
    I: float = 0.0  # noqa: E741  (integral term)

    def update(self, e: float, dt_h: float) -> float:
        u = self.Kp * e + self.I
        if not ((u >= 100 and e > 0) or (u <= 0 and e < 0)):
            self.I = float(np.clip(self.I + self.Kp * dt_h / self.Ti_h * e, -100.0, 100.0))
        return float(np.clip(self.Kp * e + self.I, 0.0, 100.0))


def replay_pi(sp: np.ndarray, t: np.ndarray, Kp: float, Ti_h: float, dt_h: float, I0: float = 0.0) -> np.ndarray:
    """Open loop: the valve % a PI would give for measured temperatures t and setpoints sp."""
    c, out = PI(Kp, Ti_h, I0), np.full(len(t), np.nan)
    for i, (s, x) in enumerate(zip(sp, t)):
        if np.isnan(s) or np.isnan(x):
            continue
        out[i] = c.update(s - x, dt_h)
    return out


def fit_pi(cfg: dict, df: pd.DataFrame, test_frac: float = 0.3) -> pd.DataFrame:
    """Fit one (Kp, Ti) per thermostat group to recorded valve % (measured sensor and setpoint in,
    valve % out). The last `test_frac` of each zone's setpoint period is held out. Returns one row
    per zone with Kp, Ti_h, train/test RMSE (% points) and the RMSE of a constant valve."""
    from scipy.optimize import least_squares
    th = cfg["controls"]["thermostats"]
    if not th:
        raise ValueError("controls.thermostats is empty")
    dt_h = step_seconds(df) / 3600
    data = {}
    for z, t in th.items():
        cols = [t["setpoint"], cfg["zones"][z]["sensor"], t["valve"]]
        missing = [c for c in cols if c not in df]
        if missing:
            raise ValueError(f"fit_pi: zone {z} needs columns {missing} (setpoint, sensor, valve %)")
        d = df[cols]
        ok = d.dropna().index
        if len(ok) < 10:
            raise ValueError(f"fit_pi: zone {z} has too little data with setpoint, sensor and valve")
        d = d.loc[ok[0]: ok[-1]]
        n_tr = int(len(d) * (1 - test_frac))
        data[z] = (d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float), d.iloc[:, 2].to_numpy(float), n_tr)
    groups: dict[str, list] = {}
    for z, t in th.items():
        groups.setdefault(t["group"], []).append(z)
    rows = []
    for g, zones in groups.items():
        def res(theta, zones=zones):
            return np.concatenate([np.nan_to_num((replay_pi(sp, t, *np.exp(theta), dt_h) - v)[:n])
                                   for sp, t, v, n in (data[z] for z in zones)])
        sol = least_squares(res, [np.log(100.0), np.log(4.0)],
                            bounds=([np.log(5), np.log(0.1)], [np.log(2000), np.log(100)]))
        Kp, Ti = np.exp(sol.x)
        for z in zones:
            sp, t, v, n = data[z]
            err = replay_pi(sp, t, Kp, Ti, dt_h) - v
            rows.append({"zone": z, "group": g, "Kp": float(Kp), "Ti_h": float(Ti),
                         "rmse_train": float(np.sqrt(np.nanmean(err[:n] ** 2))),
                         "rmse_test": float(np.sqrt(np.nanmean(err[n:] ** 2))),
                         "rmse_test_const": float(np.sqrt(np.nanmean((np.nanmean(v[:n]) - v[n:]) ** 2)))})
    return pd.DataFrame(rows).set_index("zone")


class Controller:
    """Closed-loop controller for `home_thermal.core.simulate` (pass as `controller=`).

    tuning: per thermostat zone {"Kp": .., "Ti_h": ..} (e.g. `fit_pi(...)` or its
        `.to_dict("index")`); every zone under controls.thermostats needs one.
    pump_modes: override `controls.pumps.<name>.mode`, e.g. {"UFH_A": "always"}.
    hysteresis: False to leave the hysteresis-controlled pumps at their observed values.

    A thermostat reads its zone's simulated air temperature; one on a radiator valve also reads
    the valve-sensor warming, lagged like in `sensor_temps`, as it was at the start of the step."""

    def __init__(self, net: Network, df: pd.DataFrame, tuning, pump_modes: dict | None = None,
                 hysteresis: bool = True):
        self.net, self.df = net, df
        ctl = net.cfg["controls"]
        self.dt_h = step_seconds(df) / 3600
        tuning = tuning.to_dict("index") if isinstance(tuning, pd.DataFrame) else dict(tuning)
        self.th = ctl["thermostats"]
        missing = [z for z in self.th if z not in tuning]
        if missing:
            raise KeyError(f"tuning: no Kp/Ti_h for thermostat zones {missing}")
        self.pi = {z: PI(tuning[z]["Kp"], tuning[z]["Ti_h"]) for z in self.th}
        pump_modes = pump_modes or {}
        unknown = set(pump_modes) - set(ctl["pumps"])
        if unknown:
            raise KeyError(f"pump_modes: {sorted(unknown)} not under controls.pumps")
        bad = {k: m for k, m in pump_modes.items() if m not in PUMP_MODES}
        if bad:
            raise ValueError(f"pump_modes: invalid modes {bad}; use one of {PUMP_MODES}")
        self.pumps = {k: {**v, "mode": pump_modes.get(k, v["mode"])} for k, v in ctl["pumps"].items()}
        self.hyst = ctl["hysteresis"] if hysteresis else {}
        needed = [t["setpoint"] for t in self.th.values()] + [h["setpoint"] for h in self.hyst.values()]
        needed += [p["follow"] for p in self.pumps.values() if p["mode"] == "follow"
                   and p["follow"] not in {t["valve"] for t in self.th.values()}]
        missing = [c for c in needed if c not in df]
        if missing:
            raise ValueError(f"Controller: data has no columns {missing} (setpoints / followed valves)")
        self.hyst_on = dict.fromkeys(self.hyst, False)
        self.sens: dict[str, float] = {}
        self.a = sensor_lag(net, step_seconds(df))
        self.log: list[dict] = []

    def _reading(self, z, x, row, valve):
        """What zone z's thermostat reads (see class docstring)."""
        t_air = x[self.net.iA[z]]
        r = self.net.cfg["emitters"]["radiators"].get(z)
        if r is None or not r["sensor_on_valve"]:
            return t_air
        warm = valve_sensor_warming(self.net, z, {**row, r["valve"]: valve}, t_air)
        s = self.sens.get(z, warm)
        self.sens[z] = self.a * s + (1 - self.a) * warm
        return t_air + s

    def step(self, k, x, row):
        out = {}
        for z, c in self.pi.items():
            col = self.th[z]["valve"]
            prev = row.get(col, 0.0) if not self.log else self.log[-1].get(col, 0.0)
            meas = self._reading(z, x, row, prev)
            sp = row.get(self.th[z]["setpoint"], np.nan)
            row[col] = out[col] = c.update(sp - meas, self.dt_h) if not np.isnan(sp) else 0.0
        for name, p in self.pumps.items():
            if p["mode"] == "always":
                row[name] = out[name] = 1.0
            elif p["mode"] == "follow":
                row[name] = out[name] = float(row.get(p["follow"], 0.0) > 0)
        for name, h in self.hyst.items():
            t, sp = x[self.net.iA[h["zone"]]], row.get(h["setpoint"], np.nan)
            if not np.isnan(sp):
                if t <= sp - h["below"]:
                    self.hyst_on[name] = True
                elif t >= sp + h["above"]:
                    self.hyst_on[name] = False
            row[name] = out[name] = float(self.hyst_on[name])
        self.log.append(out)

    def frame(self) -> pd.DataFrame:
        """Everything the controller wrote, per step."""
        return pd.DataFrame(self.log, index=self.df.index[: len(self.log)])

    def inputs(self) -> pd.DataFrame:
        """The data with the controller's columns in place of the measured ones: pass this to
        `sensor_temps` after a closed-loop run."""
        out = self.df.copy()
        f = self.frame()
        out.loc[f.index, f.columns] = f
        return out
