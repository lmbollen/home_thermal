"""Home Assistant recorder database -> model input DataFrame (see home_thermal.core.data).

Two sources inside the recorder DB (SQLite):
- `states`: every change at full resolution, kept ~10 days by default -> `load_states`.
- `statistics`: hourly means, kept indefinitely -> `load_statistics` (coarser).
`load_full` joins them: statistics before the states window, states after it.

Everything house-specific (entity ids, thresholds) comes from a YAML file, see `HassConfig`.
The DB is opened read-only.
"""
from __future__ import annotations

import io
import json
import sqlite3
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

STATE_VALUES = {"on": 1.0, "off": 0.0, "heat": 1.0, "idle": 0.0}
_SENTINEL = -9.99e9


@dataclass
class HassConfig:
    entities: dict                       # data column -> entity_id
    setpoints: dict = field(default_factory=dict)  # setpoint column -> climate entity_id
    setpoint_off: float = 5.0            # setpoint used while a climate entity is 'off'
    host: str | None = None              # ssh target for fetch_db, e.g. root@homeassistant.local
    ssh_key: str | None = None
    config_dir: str = "/config"
    # water circuit: column with the water flow (L/h) and the threshold for "circulating"
    flow_column: str = "water_flow"
    circulating_above: float = 100.0
    nominal_flow: float = 900.0          # statistics only: typical flow while running (L/h)
    water_sensor_offset_K: float = 0.0   # flow reads this much above return at standstill
    pv_energy: str | None = None         # energy counter column (kWh) to derive PV_W from
    pump_running_W: dict = field(default_factory=dict)   # statistics: pump col -> {power: col, running_W: W}
    household_power: dict | None = None  # {column, plus: [...], minus: [...]}
    statistics_start: str | None = None
    kw_columns: list = field(default_factory=list)   # columns whose entity reports kW in the states table

    @classmethod
    def load(cls, path: str | Path) -> HassConfig:
        try:
            with open(path, encoding="utf-8") as f:
                d = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"{path}: not valid YAML: {e}") from None
        if not isinstance(d, dict):
            raise ValueError(f"{path}: expected a mapping with at least 'entities'")
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(cls.__dataclass_fields__)}")
        if not isinstance(d.get("entities"), dict) or not d["entities"]:
            raise ValueError(f"{path}: 'entities' is required (data column -> entity_id)")
        return cls(**d)


def fetch_db(cfg: HassConfig, dest: str | Path) -> Path:
    """Copy home-assistant_v2.db (+ -wal/-shm) over ssh + tar. Read-only on the HA side."""
    if not cfg.host:
        raise ValueError("fetch_db: 'host' is not set in the Home Assistant config")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["ssh", "-o", "BatchMode=yes"]
    if cfg.ssh_key:
        cmd += ["-i", str(Path(cfg.ssh_key).expanduser())]
    files = "home-assistant_v2.db home-assistant_v2.db-wal home-assistant_v2.db-shm"
    cmd += [cfg.host, f"cd {cfg.config_dir} && tar cf - {files}"]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(out)) as tf:
        tf.extractall(dest, filter="data")
    return dest / "home-assistant_v2.db"


def _connect(db) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(db).resolve().as_posix()}?mode=ro", uri=True)


def _utc(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _to_num(s: pd.Series) -> pd.Series:
    return s.map(STATE_VALUES).fillna(pd.to_numeric(s, errors="coerce"))


def correct_water_temps(df: pd.DataFrame, offset_K: float) -> pd.DataFrame:
    """Remove a fixed flow/return sensor offset, half from each sensor (mean unchanged).
    Raw values are kept as T_flow_raw / T_return_raw. Returns a new DataFrame."""
    df = df.copy()
    df["T_flow_raw"], df["T_return_raw"] = df["T_flow"], df["T_return"]
    df["T_flow"] = df["T_flow_raw"] - offset_K / 2
    df["T_return"] = df["T_return_raw"] + offset_K / 2
    return df


def _heat_per_K(df: pd.DataFrame, min_dT: float) -> pd.Series:
    """Heat-meter output per K of RAW flow-return difference (for the meter offset correction)."""
    dT = df["T_flow"] - df["T_return"]
    return (df["HP_heat_W"] / dT).where((dT > min_dT) & (df["HP_heat_W"] > 0), 0.0)


def _has(df, *cols):
    return all(c in df for c in cols)


def load_states(db, cfg: HassConfig, step: str = "15min") -> pd.DataFrame:
    """Full-resolution states, step-held to 1 minute, then averaged to `step`."""
    con = _connect(db)
    q = """select m.entity_id, s.state, s.last_updated_ts
           from states s join states_meta m on m.metadata_id = s.metadata_id
           where m.entity_id in ({})""".format(",".join("?" * len(cfg.entities)))
    raw = pd.read_sql_query(q, con, params=list(cfg.entities.values()))
    con.close()
    if raw.empty:
        raise ValueError("load_states: none of the configured entities is in the states table")
    raw["t"] = pd.to_datetime(raw["last_updated_ts"], unit="s", utc=True)
    raw["v"] = _to_num(raw["state"])
    t0, t1 = raw["t"].min().ceil(step), raw["t"].max().floor(step)
    minute = pd.date_range(t0, t1, freq="1min")
    cols = {}
    for col, ent in cfg.entities.items():
        s = raw[raw.entity_id == ent].set_index("t")["v"].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if s.empty:
            cols[col] = pd.Series(np.nan, index=minute)
            continue
        # HA records on change: a value holds until the next record; unavailable stays NaN
        g = s.fillna(_SENTINEL).reindex(s.index.union(minute)).ffill()
        cols[col] = g.reindex(minute).replace(_SENTINEL, np.nan)
    df = pd.DataFrame(cols)
    for col in cfg.kw_columns:
        if col in df:
            df[col] = df[col] * 1000.0
    if cfg.setpoints:
        df = df.join(load_setpoints(db, cfg, minute))
    if _has(df, "T_flow", "T_return"):
        if "HP_heat_W" in df:
            df["HP_heat_per_K"] = _heat_per_K(df, 0.3)
        df = correct_water_temps(df, cfg.water_sensor_offset_K)
    if cfg.flow_column in df:
        circ = df[cfg.flow_column] > cfg.circulating_above
        if "T_flow" in df:
            tr = df["T_return"] if "T_return" in df else df["T_flow"]
            df["T_water"] = ((df["T_flow"] + tr) / 2).where(circ)
        df["circ"] = circ.astype(float)
    out = df.resample(step).mean()
    if cfg.pv_energy and cfg.pv_energy in df:
        hours = pd.Timedelta(step).total_seconds() / 3600
        e_end = df[cfg.pv_energy].resample(step).last()
        out["PV_W"] = ((e_end - e_end.shift(1)) * 1000 / hours).clip(lower=0)
    return out


def load_setpoints(db, cfg: HassConfig, minute: pd.DatetimeIndex) -> pd.DataFrame:
    """Thermostat setpoints (the 'temperature' attribute of climate entities) on a 1-minute grid;
    state 'off' -> cfg.setpoint_off."""
    con = _connect(db)
    q = """select m.entity_id, s.state, s.last_updated_ts, a.shared_attrs
           from states s join states_meta m on m.metadata_id = s.metadata_id
           left join state_attributes a on a.attributes_id = s.attributes_id
           where m.entity_id in ({})""".format(",".join("?" * len(cfg.setpoints)))
    raw = pd.read_sql_query(q, con, params=list(cfg.setpoints.values()))
    con.close()
    raw["t"] = pd.to_datetime(raw["last_updated_ts"], unit="s", utc=True)

    def sp(row):
        if row["state"] == "off":
            return cfg.setpoint_off
        try:
            return float(json.loads(row["shared_attrs"]).get("temperature"))
        except (TypeError, ValueError, AttributeError):
            return np.nan
    raw["v"] = raw.apply(sp, axis=1) if len(raw) else np.nan
    out = {}
    for col, ent in cfg.setpoints.items():
        s = raw[raw.entity_id == ent].set_index("t")["v"].sort_index().dropna()
        s = s[~s.index.duplicated(keep="last")]
        out[col] = (s.reindex(s.index.union(minute)).ffill().reindex(minute) if len(s)
                    else pd.Series(np.nan, index=minute))
    return pd.DataFrame(out, index=minute)


def load_statistics(db, cfg: HassConfig, start=None, end=None, step: str = "15min") -> pd.DataFrame:
    """Model inputs from the hourly long-term statistics, spread over `step`.

    Coarser than `load_states`: pump states become hourly on-fractions from plug power
    (`pump_running_W`), and the water temperature is an hourly mean that includes idle minutes."""
    con = _connect(db)
    q = """select m.statistic_id e, m.unit_of_measurement u, s.start_ts t, s.mean v, s.min vmin, s.sum vsum
           from statistics s join statistics_meta m on m.id = s.metadata_id
           where m.statistic_id in ({})""".format(",".join("?" * len(cfg.entities)))
    raw = pd.read_sql_query(q, con, params=list(cfg.entities.values()))
    con.close()
    if raw.empty:
        raise ValueError("load_statistics: none of the configured entities has statistics")
    raw["t"] = pd.to_datetime(raw["t"], unit="s", utc=True)
    raw["col"] = raw["e"].map({v: k for k, v in cfg.entities.items()})
    raw.loc[raw.u == "kW", ["v", "vmin"]] *= 1000.0       # power statistics may be stored in kW
    piv = {k: raw.pivot_table(index="t", columns="col", values=k, aggfunc="mean").sort_index()
           for k in ("v", "vmin", "vsum")}
    h = piv["v"]
    if start is not None:
        h = h[h.index >= _utc(start)]
    if end is not None:
        h = h[h.index < _utc(end)]
    if h.empty:
        raise ValueError(f"load_statistics: no statistics between {start} and {end}")
    hmin, hsum = piv["vmin"].reindex(h.index), piv["vsum"].reindex(h.index)
    for pump, spec in cfg.pump_running_W.items():
        if spec["power"] in h:
            h[pump] = (h[spec["power"]] / spec["running_W"]).clip(0, 1)
    if cfg.flow_column in h:
        # if the hourly minimum flow stayed above the threshold the pump ran all hour
        full = hmin[cfg.flow_column] > cfg.circulating_above
        h["circ"] = np.where(full, 1.0, (h[cfg.flow_column] / cfg.nominal_flow).clip(0, 1))
    if _has(h, "T_flow", "T_return"):
        if "HP_heat_W" in h:
            h["HP_heat_per_K"] = _heat_per_K(h, 0.5)
        h = correct_water_temps(h, cfg.water_sensor_offset_K)
        if "circ" in h:
            h["T_water"] = ((h["T_flow"] + h["T_return"]) / 2).where(h["circ"] > 0.05)
    if cfg.pv_energy and cfg.pv_energy in hsum:
        h["PV_W"] = (hsum[cfg.pv_energy] - hsum[cfg.pv_energy].shift(1)).clip(lower=0) * 1000.0
    # each hourly row covers [t, t+1h): spread it over that hour only; missing hours stay NaN
    h = h.reindex(pd.date_range(h.index[0], h.index[-1], freq="1h"))
    idx = pd.date_range(h.index[0], h.index[-1] + pd.Timedelta("1h") - pd.Timedelta(step), freq=step)
    per_hour = int(pd.Timedelta("1h") / pd.Timedelta(step))
    return h.reindex(idx, method="ffill", limit=max(per_hour - 1, 0)) if per_hour > 1 else h.reindex(idx)


def household_power(df: pd.DataFrame, spec: dict) -> pd.Series:
    """Sum of `plus` columns minus `minus` columns, clipped at 0 (e.g. grid + PV - heating)."""
    h = df[[c for c in spec["plus"] + spec["minus"] if c in df]].fillna(0)
    total = sum(h[c] for c in spec["plus"] if c in h) - sum(h[c] for c in spec["minus"] if c in h)
    return total.clip(lower=0)


def load_full(db, cfg: HassConfig, step: str = "15min") -> pd.DataFrame:
    """Statistics before the states window (from `statistics_start`), full-resolution states after.
    Where a states value is missing (before an entity's first change in the window, or while it was
    unavailable) and an hourly statistic exists, the statistic fills it.
    Adds `from_stats` (1 where a row comes from statistics)."""
    st = load_states(db, cfg, step)
    if cfg.statistics_start is None:
        df = st
    else:
        hs = load_statistics(db, cfg, start=cfg.statistics_start, step=step)
        cols = [c for c in st.columns if c in hs.columns]
        idx = pd.date_range(min(hs.index[0], st.index[0]), st.index[-1], freq=step)
        df = st.reindex(idx)
        df["from_stats"] = (idx < st.index[0]).astype(float)
        df[cols] = df[cols].combine_first(hs[cols].reindex(idx))
    if cfg.household_power:
        df[cfg.household_power["column"]] = household_power(df, cfg.household_power)
    if not (df.index.to_series().diff().iloc[1:] == pd.Timedelta(step)).all():
        raise ValueError("load_full: irregular time grid")
    return df
