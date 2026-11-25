"""Heat inputs per step: emitters (radiators, floor heating, heat-source unit), hot-water tank,
sun and background gain. Plus what each room sensor is expected to read.

Functions take a `row`: a dict of data columns for one step (see `data.py` for the column names).
None of them modify their inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import Network

CP_WATER = 4186.0   # J/kgK
DT50 = 49.83        # EN 442 log-mean temperature difference at 75/65/20 C


def valve_value(r: dict, data) -> float | np.ndarray:
    """Valve opening (%) of radiator r: its data column, or the fixed number in the config."""
    v = r["valve"]
    if isinstance(v, str):
        return data.get(v, 0.0) if isinstance(data, dict) else data[v].to_numpy(float)
    return v if isinstance(data, dict) else np.full(len(data), float(v))


def radiator_heat(net: Network, r: dict, valve, row, t_air, zone: str | None = None):
    """Radiator output (W). Works on scalars (row = dict) or arrays (row = DataFrame).

    ntu: flow-limited heat exchanger. Water enters at T_flow; flow = m_flow * m_flow_<zone> * design flow *
         max(0, (valve - v0)/(100 - v0)); Q = Cdot (T_in - T_air)(1 - exp(-UA/Cdot)).
         Small openings: limited by flow (thermostatic-valve behaviour). Large openings: limited by
         the radiator surface (catalogue q50 curve, scaled by m_rad). m_flow_<zone> (default 1)
         is this radiator's own share of the water: how the circuit's flow divides between them.
    catalogue: Q = m_rad q50 (dT/49.83)^n (valve/100)^gamma_v at the mean water temperature."""
    p = net.p
    v = np.clip(np.nan_to_num(np.asarray(valve, float)), 0.0, 100.0)
    circ = np.nan_to_num(np.asarray(row["circ"], float))
    if net.cfg["emitters"]["radiator_model"] == "ntu":
        dT = np.nan_to_num(np.asarray(row["T_flow"], float) - t_air)
        frac = np.clip((v - p["v0"]) / (100.0 - p["v0"]), 0.0, 1.0)
        flow = r.get("flow_lph", r["q50"] / (CP_WATER * 10.0) * 3600.0)     # L/h at 100 %, 10 K drop
        own = p.get(f"m_flow_{zone}", 1.0) if zone else 1.0
        cdot = p["m_flow"] * own * flow / 3600.0 * CP_WATER * frac          # W/K
        ua = p["m_rad"] * r["q50"] / DT50 ** r["n"] * np.maximum(dT, 0.5) ** (r["n"] - 1.0)
        eff = np.where(cdot > 0, cdot * (1.0 - np.exp(-ua / np.maximum(cdot, 1e-9))), 0.0)
    else:
        dT = np.nan_to_num(np.asarray(row["T_water"], float) - t_air)
        eff = (p["m_rad"] * r["q50"] / DT50 ** r["n"] * np.maximum(dT, 1e-9) ** (r["n"] - 1.0)
               * (v / 100.0) ** p["gamma_v"])
    q = np.where((dT > 0) & (circ > 0), eff * dT * circ, 0.0)
    return float(q) if q.ndim == 0 else q


def pump_fraction(net: Network, pump: str, row: dict) -> float:
    """Effective opening of a floor-heating circuit: 1 for 'always', else the pump column (0..1)
    plus a fitted residual flow while it is off."""
    if pump == "always":
        return 1.0
    flag = row.get(pump, 0.0)
    flag = 0.0 if flag is None or np.isnan(flag) else min(max(flag, 0.0), 1.0)
    return flag + (1.0 - flag) * net.p.get(f"f_off_{pump}", 0.0)


def emitter_heat(net: Network, x: np.ndarray, row: dict) -> tuple[np.ndarray, float]:
    """Heat into each node from the water circuit and the hot-water tank (W), and the total heat
    drawn from the heat source (W)."""
    em, p = net.cfg["emitters"], net.p
    q = np.zeros(net.n)
    tank_in, tap, extracted = dhw_heat(net, x, row)
    if net.iT is not None:
        q[net.iT] = tank_in - tap
    tw, circ = row.get("T_water", np.nan), row.get("circ", 0.0)
    if not (circ > 0) or tw is None or np.isnan(tw):
        return q, extracted
    q_source = 0.0
    for z, r in em["radiators"].items():
        h = radiator_heat(net, r, valve_value(r, row), row, x[net.iA[z]], z)
        q[net.iA[z]] += h
        q_source += h
    for z, u in em["floor_heating"].items():
        s = net.iS[z]
        cap = row.get(f"t_max:{z}")
        tw_z = tw if cap is None or cap != cap else min(tw, cap)    # mixing valve / limiter setting
        h = p["m_ufh"] * u["K_ws"] * u["area"] * (tw_z - x[s]) * pump_fraction(net, u["pump"], row) * circ
        q[s] += h
        q_source += h
    for z in em["pipes"]:
        h = p[f"UA_pipe_{z}"] * max(tw - x[net.iA[z]], 0.0) * circ
        q[net.iA[z]] += h
        q_source += h
    hz, ua = source_zone(net.cfg)
    if hz:
        h = p[ua] * (tw - x[net.iA[hz]]) * circ
        q[net.iA[hz]] += h
        q_source += h
    return q, q_source + extracted + outdoor_pipe_loss(net, row)


def floor_caps(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """df with a column `t_max:<zone>` per floor circuit with a `t_max` (its mixing valve or limiter
    setting over time; NaN = no cap). Returns df itself if no circuit has one."""
    caps = {z: u["t_max"] for z, u in cfg["emitters"]["floor_heating"].items() if "t_max" in u}
    if not caps:
        return df
    df = df.copy()
    for z, v in caps.items():
        if isinstance(v, (int, float)):
            df[f"t_max:{z}"] = float(v)
            continue
        col = pd.Series(np.nan, index=df.index)
        for t, c in v:
            t = t if df.index.tz is not None else t.tz_localize(None)
            col[df.index >= t] = c
        df[f"t_max:{z}"] = col
    return df


def outdoor_pipe_loss(net: Network, row: dict) -> float:
    """Heat lost by the water pipes outdoors (monoblock heat pump), W: UA_pipe_out x (mean water
    temperature - outdoor) while water circulates. The heat pump produces (and its meter counts)
    this heat, but it never reaches a zone. 0 without heat_pump.outdoor_pipes."""
    ua = net.p.get("UA_pipe_out")
    tw, circ, t_out = row.get("T_water", np.nan), row.get("circ", 0.0), row.get("T_out", np.nan)
    if not ua or not (circ > 0) or tw is None or np.isnan(tw) or t_out is None or np.isnan(t_out):
        return 0.0
    return float(max(ua * (tw - t_out), 0.0) * circ)


def source_zone(cfg: dict) -> tuple[str | None, str | None]:
    """Zone where the heat source (heat pump indoor unit or boiler) loses heat, and its parameter."""
    if cfg["boiler"] is not None:
        return cfg["boiler"]["zone"], "UA_boiler"
    return cfg["heat_pump"].get("zone"), "UA_hp"


def dhw_heat(net: Network, x: np.ndarray, row: dict) -> tuple[float, float, float]:
    """Water/water hot-water heat pump taking its source heat from the heating circuit.
    Returns (heat into the tank, tap draw, heat extracted from the heating circuit), all W.
    Without a `dhw` section all are 0."""
    if net.iT is None:
        return 0.0, 0.0, 0.0
    d, p = net.cfg["dhw"], net.p
    pe = row.get(d["power"], 0.0)
    pe = 0.0 if (pe is None or np.isnan(pe) or pe < 0) else pe
    t_tank = x[net.iT]
    on = pe > d["on_threshold_W"] and t_tank < d["T_max"]
    tap = p["tap_W"] if t_tank > d["T_cold"] + 5 else 0.0
    cop = p["cop_dhw"]
    return (cop * pe, tap, (cop - 1.0) * pe) if on else (0.0, tap, 0.0)


def solar_gain(net: Network, df: pd.DataFrame) -> np.ndarray:
    """Sun through windows per step and node (N x n): g_sol x solar proxy column, spread over
    windows by area x orientation weight. Zero if the proxy column is absent."""
    proxy = net.cfg["solar"]["proxy"]
    if proxy not in df or "g_sol" not in net.p:
        return np.zeros((len(df), net.n))
    sun = np.clip(np.nan_to_num(df[proxy].to_numpy(float)), 0, None) * net.p["g_sol"]
    return np.outer(sun, net.solar_share)


def background_gain(net: Network, df: pd.DataFrame) -> np.ndarray:
    """Background gain per step and node (N x n): Q_int (W), or with background_gain.mode 'meter'
    Q_int plus a fixed share of a household electricity column; spread over occupied zones by area."""
    bg = net.cfg["background_gain"]
    gain = np.full(len(df), net.p["Q_int"])
    if bg["mode"] == "meter" and bg["meter_column"] in df:
        gain = gain + bg["meter_heat_share"] * np.clip(np.nan_to_num(df[bg["meter_column"]].to_numpy(float)), 0, None)
    return np.outer(gain, net.int_share)


def free_gains(net: Network, df: pd.DataFrame) -> np.ndarray:
    """Sun + background gain per step and node (N x n)."""
    return solar_gain(net, df) + background_gain(net, df)


# ------------------------------------------------------------------ sensors

def valve_sensor_warming(net: Network, z: str, row, t_air):
    """How much warmer than the air a valve-mounted sensor in zone z reads right now (K), before
    any time lag. Scalars (row = dict) or arrays (row = DataFrame). 0 for other sensors.
    ntu: kappa_q K per 100 W of radiator output. catalogue: kappa_s x opening x water excess."""
    r = net.cfg["emitters"]["radiators"].get(z)
    if r is None or not r["sensor_on_valve"]:
        return 0.0 * np.asarray(t_air)
    v = valve_value(r, row)
    if net.cfg["emitters"]["radiator_model"] == "ntu":
        return net.p["kappa_q"] * radiator_heat(net, r, v, row, t_air, z) / 100.0
    excess = np.nan_to_num(np.clip(np.asarray(row["T_water"], float) - t_air, 0, None))
    return (net.p["kappa_s"] * np.clip(np.nan_to_num(np.asarray(v, float)) / 100.0, 0, 1) * excess
            * np.nan_to_num(np.asarray(row["circ"], float)))


def sensor_lag(net: Network, dt_s: float) -> float:
    """Per-step memory of the valve-sensor lag (ntu only; the catalogue model has none)."""
    if net.cfg["emitters"]["radiator_model"] != "ntu":
        return 0.0
    return float(np.exp(-dt_s / 3600 / net.cfg["emitters"]["valve_sensor_tau_h"]))


def sensor_reading(net: Network, z: str, t_air: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Predicted reading of zone z's sensor over df. A valve-mounted sensor reads air temperature
    plus its warming, passed through a first-order lag (the sensor body; valve_sensor_tau_h)."""
    r = net.cfg["emitters"]["radiators"].get(z)
    if r is None or not r["sensor_on_valve"]:
        return t_air
    target = t_air + valve_sensor_warming(net, z, df, t_air)
    if net.cfg["emitters"]["radiator_model"] != "ntu" or len(df) < 2:
        return target                      # no sensor lag (catalogue model, or a single step)
    a = sensor_lag(net, (df.index[1] - df.index[0]).total_seconds())
    out = np.empty_like(target, dtype=float)
    s = target[0]
    for i, tg in enumerate(target):
        s = a * s + (1 - a) * tg
        out[i] = s
    return out


def sensor_temps(net: Network, X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Predicted reading of every zone sensor (columns = sensor column names). For a closed-loop
    run, pass the inputs the controller used (`Controller.inputs()`), not the measured ones."""
    return pd.DataFrame({zc["sensor"]: sensor_reading(net, z, X[f"air:{z}"].to_numpy(), df)
                         for z, zc in net.cfg["zones"].items() if zc["sensor"]}, index=X.index)
