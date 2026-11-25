"""Run the network over a time series of measured inputs."""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from .data import check_inputs, prepare, step_seconds
from .network import Network
from .physics import background_gain, dhw_heat, emitter_heat, floor_caps, outdoor_pipe_loss, solar_gain


class StepController(Protocol):
    """Anything that may overwrite inputs (valve %, pump flags) in `row` before each step."""
    def step(self, k: int, x: np.ndarray, row: dict) -> None: ...


def simulate(net: Network, df: pd.DataFrame, x0: np.ndarray | None = None,
             controller: StepController | None = None, q_extra=None, record: bool = False):
    """Simulate over df (regular time grid). Returns (node states DataFrame, heat drawn from the
    heat source Series, W).

    x0: initial state (default: from the sensors at the start of df, see `initial_state`).
    controller: replaces measured valve/pump columns step by step (closed loop), e.g.
        `home_thermal.control.Controller`.
    q_extra: extra heat per node, W (n-vector or N x n), e.g. an estimated disturbance.
    record: also return a dict with every heat input per step and node (for `energy_balance`)."""
    df = prepare(df)
    check_inputs(net.cfg, df)
    df = floor_caps(net.cfg, df)
    dt = step_seconds(df)
    Ad, Bb, Bq = net.discretise(dt)
    N = len(df)
    X, Qsrc, Qe = np.zeros((N, net.n)), np.zeros(N), np.zeros((N, net.n))
    tank = np.zeros((N, 3))                          # into tank, tap draw, extracted
    pipes = np.zeros(N)                              # lost by the outdoor pipes
    sun, bg = solar_gain(net, df), background_gain(net, df)
    gains = sun + bg + (0.0 if q_extra is None else q_extra)
    tb = np.array([net.boundary_temps(t) for t in df["T_out"].ffill().bfill().to_numpy()])
    x = initial_state(net, df) if x0 is None else np.asarray(x0, float)
    for k, row in enumerate(df.to_dict("records")):
        X[k] = x
        if controller is not None:
            controller.step(k, x, row)
        Qe[k], Qsrc[k] = emitter_heat(net, x, row)
        if record and net.iT is not None:
            tank[k] = dhw_heat(net, x, row)
        if record:
            pipes[k] = outdoor_pipe_loss(net, row)
        x = Ad @ x + Bb @ tb[k] + Bq @ (Qe[k] + gains[k])
    states = pd.DataFrame(X, index=df.index, columns=net.names)
    heat = pd.Series(Qsrc, index=df.index)
    if not record:
        return states, heat
    return states, heat, dict(Qe=Qe, Qs=sun, Qi=bg, Qx=np.zeros_like(sun) if q_extra is None else gains - sun - bg,
                              tank=tank, pipes=pipes, Tb=tb, dt=dt, index=df.index)


def initial_state(net: Network, df) -> np.ndarray:
    """Start every zone at its sensor's first reading within the first 6 hours of df (zones
    without one: mean of the others), mass at the same temperature, floor slabs 1 K warmer, the
    tank at its zone temperature (or column T_tank0). df may also be a single row (Series/dict)."""
    if isinstance(df, pd.DataFrame):
        head = df[df.index < df.index[0] + pd.Timedelta("6h")]
        first = lambda col: head[col].dropna().iloc[0] if col in head and head[col].notna().any() else np.nan  # noqa: E731
        t_tank0 = first("T_tank0")
    else:
        first = lambda col: df.get(col, np.nan) if df.get(col) is not None else np.nan  # noqa: E731
        t_tank0 = first("T_tank0")
    meas = {z: float(first(zc["sensor"])) if zc["sensor"] else np.nan for z, zc in net.cfg["zones"].items()}
    known = [v for v in meas.values() if not np.isnan(v)]
    if not known:
        raise ValueError("initial_state: no zone sensor has a value in the first 6 hours of the data")
    fallback = float(np.mean(known))
    x = np.zeros(net.n)
    for z in net.zones:
        t = fallback if np.isnan(meas[z]) else meas[z]
        x[net.iA[z]] = x[net.iM[z]] = t
    for z in net.ufh_zones:
        x[net.iS[z]] = x[net.iA[z]] + 1.0
    if net.iT is not None:
        x[net.iT] = t_tank0 if not np.isnan(t_tank0) else x[net.iA[net.cfg["dhw"]["zone"]]]
    return x
