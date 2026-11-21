"""Synthetic data for trying the library and for tests: weather, a heating schedule with simple
thermostats, and the sensor readings a house with known 'true' parameters would produce.
Works for any house configuration.

    from home_thermal.core.synthetic import demo_house, synthetic_data
    cfg = demo_house()
    df = synthetic_data(cfg, {"m_env": 1.3}, days=10)
"""
from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd
import yaml

from .boiler import Boiler
from .config import normalise
from .heatpump import HeatPump
from .network import Network
from .physics import sensor_temps
from .simulate import simulate


def demo_house_yaml() -> str:
    """The bundled demo house configuration as YAML text."""
    return files("home_thermal.core").joinpath("examples/demo_house.yaml").read_text(encoding="utf-8")


def demo_house() -> dict:
    """The bundled demo house configuration (normalised)."""
    return normalise(yaml.safe_load(demo_house_yaml()))


class _Thermostats:
    """Proportional valves toward a comfort/setback schedule, pumps and circulation on demand."""

    def __init__(self, net: Network, comfort: np.ndarray, rng, no_heating: np.ndarray):
        self.net, self.comfort, self.rng, self.no_heating = net, comfort, rng, no_heating
        cfg = net.cfg
        self.rad = {z: r["valve"] for z, r in cfg["emitters"]["radiators"].items() if isinstance(r["valve"], str)}
        for z, t in cfg["controls"]["thermostats"].items():     # thermostats driving other columns
            self.rad.setdefault(z, t["valve"])
        self.follow = {k: v["follow"] for k, v in cfg["controls"]["pumps"].items() if v["mode"] == "follow"}
        self.pumps = {}
        for z, u in cfg["emitters"]["floor_heating"].items():
            if u["pump"] != "always":
                self.pumps.setdefault(u["pump"], []).append(z)
        self.ufh_always = [z for z, u in cfg["emitters"]["floor_heating"].items() if u["pump"] == "always"]
        self.log = []

    def step(self, k, x, row):
        net, out = self.net, {}
        sp = 20.0 if self.comfort[k] else 17.0
        demand = []
        for z, col in self.rad.items():
            e = sp - x[net.iA[z]]
            v = float(np.clip(40 * e + 20 + 5 * self.rng.standard_normal(), 0, 100))
            row[col] = out[col] = v
            demand.append(v > 25)
        for pump, zones in self.pumps.items():
            if pump in self.follow and self.follow[pump] in row:
                on = float(row[self.follow[pump]] > 0)
            else:
                on = float(any(x[net.iA[z]] < sp for z in zones))
            row[pump] = out[pump] = on
            demand.append(on > 0)
        demand += [x[net.iA[z]] < sp + 0.3 for z in self.ufh_always]
        row["circ"] = out["circ"] = float((any(demand) or not demand) and not self.no_heating[k])
        self.log.append(out)


def synthetic_data(cfg: dict, true_params: dict | None = None, days: float = 10, step: str = "15min",
                   start: str = "2026-01-05", seed: int = 0, noise_K: float = 0.05,
                   solar: bool = True, heat_pump: bool = True) -> pd.DataFrame:
    """Simulate the house with `true_params` (others at their prior) under cold weather, with
    simple thermostats aiming at 20 C (comfort hours) / 17 C, and return a DataFrame in the input
    format, sensors included (plus noise_K measurement noise).

    solar: include a sunshine proxy column (cfg solar.proxy, W) and sun.
    heat_pump: include the heat-source measurements: for a house with a `boiler`, its gas column
        (with eta_boiler from true_params, plus tap-water gas in morning showers, flagged in the
        dhw_flag column if one is configured); otherwise HP_heat_W and HP_elec_W (eta 0.5)."""
    rng = np.random.default_rng(seed)
    n = int(pd.Timedelta(f"{days}D") / pd.Timedelta(step))
    if n < 2:
        raise ValueError("synthetic_data: needs at least two steps")
    idx = pd.date_range(start, periods=n, freq=step, tz="UTC")
    hours = (idx - idx[0]).total_seconds().to_numpy() / 3600
    hod = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    t_out = 2 + 4 * np.sin(2 * np.pi * hours / (24 * 5)) + 3 * np.sin(2 * np.pi * (hod - 9) / 24)
    t_out += rng.normal(0, 0.3, n).cumsum() * 0.02
    df = pd.DataFrame({"T_out": t_out}, index=idx)
    if solar:
        sun = np.clip(np.sin(np.pi * (hod - 8) / 8), 0, None) * 400 * (rng.random(n) ** 0.5)
        sun[(hod < 8) | (hod > 16)] = 0
        df[cfg["solar"]["proxy"]] = sun
    comfort = ((hod >= 6.5) & (hod < 9)) | ((hod >= 17) & (hod < 23))
    for t in cfg["controls"]["thermostats"].values():
        df[t["setpoint"]] = np.where(comfort, 20.0, 17.0)
    em = cfg["emitters"]
    df["T_flow"] = np.clip(35 - 0.6 * (df["T_out"] - 5), 28, 45) - (8.0 if not em["radiators"] else 0.0)
    df["T_water"] = df["T_flow"] - 2.5
    df["T_return"] = df["T_flow"] - 5.0
    df["circ"] = 1.0
    for col in {r["valve"] for r in em["radiators"].values() if isinstance(r["valve"], str)} | {
            t["valve"] for t in cfg["controls"]["thermostats"].values()}:
        df[col] = 0.0
    for u in em["floor_heating"].values():
        if u["pump"] != "always":
            df[u["pump"]] = 0.0
    if cfg["dhw"]:
        df[cfg["dhw"]["power"]] = np.where((hod >= 2) & (hod < 3), 500.0, 0.0)
    for zc in cfg["zones"].values():
        if zc["sensor"]:
            df[zc["sensor"]] = 19.0          # start of the run: everything at 19 C
    net = Network(cfg, dict(true_params or {}))
    # combi boiler: a 15-min morning shower, during which it does not heat the house
    shower = ((hod >= 7) & (hod < 7.25)).astype(float)
    tap = shower if (heat_pump and cfg["boiler"] is not None) else np.zeros(n)
    thermostats = _Thermostats(net, comfort, rng, tap > 0)
    X, Q = simulate(net, df, controller=thermostats)
    log = pd.DataFrame(thermostats.log, index=idx)
    df[log.columns] = log
    S = sensor_temps(net, X, df)
    for c in S.columns:
        df[c] = S[c] + rng.normal(0, noise_K, n)
    if heat_pump and cfg["boiler"] is not None:
        b = cfg["boiler"]
        fuel = Boiler.from_config(cfg, net.p).fuel(Q) + tap * 20000.0      # combi boiler on tap water
        step_h = pd.Timedelta(step).total_seconds() / 3600
        per_step = {"W": 1.0, "kWh": step_h / 1000, "m3": step_h / 1000 / b["kwh_per_m3"]}[b["gas_unit"]]
        df[b["gas_column"]] = fuel * per_step
        if b["dhw_flag"]:
            df[b["dhw_flag"]] = tap
    elif heat_pump:
        df["HP_heat_W"] = Q
        df["HP_elec_W"] = HeatPump(eta=0.5).electric(Q, df["T_out"], df["T_flow"])
    return df
