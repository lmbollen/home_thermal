"""The input data format: one pandas DataFrame on a regular UTC time grid (e.g. 15 min).

Fixed columns (the model's own names):

    T_out       outdoor temperature (C)                                   required
    T_flow      water temperature leaving the heat source (C)             required with emitters
    T_return    water temperature returning to it (C)                     optional
    T_water     mean water temperature while circulating (C); derived from T_flow/T_return if absent
    circ        fraction of the step that water circulated (0..1)         required with emitters
    HP_heat_W   heat-meter output of the heat source (W)                  for calibration
    HP_elec_W   electricity used by the heat source (W)                   for calibration

Columns named in the house configuration: zone sensors (`zones.<z>.sensor`), radiator valve %
(`emitters.radiators.<z>.valve`), pump on-fractions (`emitters.floor_heating.<z>.pump`), the solar
proxy (`solar.proxy`, default PV_W), the hot-water power (`dhw.power`), thermostat setpoints
(`controls.thermostats.<z>.setpoint`). Any data source (Home Assistant, CSV, a BMS export) only
has to produce these columns.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def step_seconds(df: pd.DataFrame) -> float:
    return (df.index[1] - df.index[0]).total_seconds()


def required_columns(cfg: dict) -> list[str]:
    """Columns the simulation needs for this house (calibration additionally needs sensors)."""
    em = cfg["emitters"]
    cols = ["T_out"]
    if em["radiators"] or em["floor_heating"] or cfg["heat_pump"].get("zone"):
        cols += ["T_water", "circ"]
        if em["radiators"] and em["radiator_model"] == "ntu":
            cols.append("T_flow")
    cols += [r["valve"] for r in em["radiators"].values() if isinstance(r["valve"], str)]
    cols += [u["pump"] for u in em["floor_heating"].values() if u["pump"] != "always"]
    if cfg["dhw"]:
        cols.append(cfg["dhw"]["power"])
    return list(dict.fromkeys(cols))


def check_inputs(cfg: dict, df: pd.DataFrame) -> None:
    """Raise a clear error if df is not usable for this house."""
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
        raise ValueError("data: needs a DatetimeIndex with at least two rows")
    d = np.diff(df.index.asi8)
    if not (d == d[0]).all() or d[0] <= 0:
        raise ValueError("data: the time grid must be regular and increasing (resample first)")
    missing = [c for c in required_columns(cfg) if c not in df]
    if missing:
        raise ValueError(f"data: missing columns {missing} (see docs/data.md for the format)")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Derive T_water from T_flow/T_return where it is missing (only while water circulates).
    Returns df itself when there is nothing to derive."""
    if "T_water" in df or "T_flow" not in df:
        return df
    df = df.copy()
    tr = df["T_return"] if "T_return" in df else df["T_flow"]
    circ = df["circ"] if "circ" in df else pd.Series(1.0, index=df.index)
    df["T_water"] = ((df["T_flow"] + tr) / 2).where(circ > 0)
    return df


def check_sensors(cfg: dict, df: pd.DataFrame) -> None:
    """Raise if a configured zone sensor column is missing (needed for calibration)."""
    missing = [zc["sensor"] for zc in cfg["zones"].values() if zc["sensor"] and zc["sensor"] not in df]
    if missing:
        raise ValueError(f"data: sensor columns {missing} are configured but not in the data")


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV in this format (first column = timestamp, UTC)."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return prepare(df)
