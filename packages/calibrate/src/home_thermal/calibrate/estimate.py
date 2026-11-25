"""State estimation for forecasting: what the twin does with heat it cannot know about.

Doors, windows, people, cooking: none of it is measured per room. Instead of a parameter for each,
every measured zone gets one extra state d_z (W, "unexplained heat"), a slow random walk. A Kalman
filter on the linear network estimates the node temperatures and d_z from the sensors.

This is for forecasting and diagnosis only; it is not part of the house model. The noise levels
are fixed engineering choices; tuning them on the same data would be one more way to overfit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from home_thermal.core import (
    Network,
    emitter_heat,
    floor_caps,
    free_gains,
    initial_state,
    prepare,
    sensor_temps,
    simulate,
    step_seconds,
    valve_sensor_warming,
)


@dataclass
class FilterNoise:
    sigma_d: float = 20.0      # W per sqrt(hour): how fast unexplained heat can change
    sigma_air: float = 0.10    # K per step, air nodes
    sigma_mass: float = 0.01   # K per step, mass / slab / tank nodes
    sigma_y: float = 0.15      # K, sensor noise


def measured(net: Network) -> list[tuple[str, str]]:
    return [(z, zc["sensor"]) for z, zc in net.cfg["zones"].items() if zc["sensor"]]


def kalman(net: Network, df: pd.DataFrame, noise: FilterNoise | None = None):
    """Causal filter over df. Returns (posterior node states, unexplained heat per zone W).
    Row k uses measurements up to and including k.

    Valve-mounted sensors are predicted as air + their instantaneous warming; the sensor lag of
    the simulation (valve_sensor_tau_h) is not a filter state, a small approximation."""
    nz = noise or FilterNoise()
    df = floor_caps(net.cfg, prepare(df))
    dt = step_seconds(df)
    Ad, Bb, Bq = net.discretise(dt)
    meas = measured(net)
    ia = [net.iA[z] for z, _ in meas]
    n, m = net.n, len(meas)
    F = np.zeros((n + m, n + m))
    F[:n, :n] = Ad
    F[:n, n:] = Bq[:, ia]
    F[n:, n:] = np.eye(m)
    qd = np.full(n, nz.sigma_mass ** 2)
    qd[[net.iA[z] for z in net.zones]] = nz.sigma_air ** 2
    Q = np.diag(np.concatenate([qd, np.full(m, nz.sigma_d ** 2 * dt / 3600)]))
    H = np.zeros((m, n + m))
    H[np.arange(m), ia] = 1.0
    R = nz.sigma_y ** 2
    x = np.concatenate([initial_state(net, df), np.zeros(m)])
    P = np.diag(np.concatenate([np.full(n, 1.0), np.full(m, 100.0 ** 2)]))
    gains = free_gains(net, df)
    tout = df["T_out"].ffill().bfill().to_numpy()
    Y = df[[c for _, c in meas]].to_numpy(float)
    XS, DS = np.zeros((len(df), n)), np.zeros((len(df), m))
    for k, row in enumerate(df.to_dict("records")):
        yhat = x[ia] + np.array([valve_sensor_warming(net, z, row, x[i]) for (z, _), i in zip(meas, ia)])
        ok = ~np.isnan(Y[k])
        if ok.any():
            Hk = H[ok]
            S = Hk @ P @ Hk.T + R * np.eye(ok.sum())
            K = np.linalg.solve(S, Hk @ P).T
            x = x + K @ (Y[k, ok] - yhat[ok])
            P = (np.eye(n + m) - K @ Hk) @ P
        XS[k], DS[k] = x[:n], x[n:]
        q, _ = emitter_heat(net, x[:n], row)
        xn = Ad @ x[:n] + Bb @ net.boundary_temps(tout[k]) + Bq @ (q + gains[k]) + Bq[:, ia] @ x[n:]
        x = np.concatenate([xn, x[n:]])
        P = F @ P @ F.T + Q
    return (pd.DataFrame(XS, index=df.index, columns=net.names),
            pd.DataFrame(DS, index=df.index, columns=[z for z, _ in meas]))


def forecast_errors(net: Network, df: pd.DataFrame, origins, horizons_h=(1, 6, 24), filt=None) -> pd.DataFrame:
    """Forecast every zone sensor from each origin, three ways:
    'twin+filter' (filtered state, unexplained heat held), 'twin+filter, d=0' (filtered state, no
    unexplained heat) and 'persistence' (no change). One row per (origin, horizon, zone, method).
    Horizons are rounded to whole data steps; the 'horizon_steps' column says what was used."""
    df = prepare(df)
    XS, DS = filt if filt is not None else kalman(net, df)
    meas = measured(net)
    per_h = 3600.0 / step_seconds(df)
    steps = {h: max(1, int(round(h * per_h))) for h in horizons_h}
    hmax = max(steps.values())
    rows = []
    for t0 in origins:
        k0 = df.index.get_loc(t0)
        seg = df.iloc[k0: k0 + hmax + 1]
        if len(seg) < hmax + 1:
            continue
        for method in ("twin+filter", "twin+filter, d=0"):
            qx = np.zeros(net.n)
            if method == "twin+filter":
                qx[[net.iA[z] for z, _ in meas]] = DS.iloc[k0].to_numpy()
            X, _ = simulate(net, seg, x0=XS.iloc[k0].to_numpy(), q_extra=qx)
            S = sensor_temps(net, X, seg)
            for h, i in steps.items():
                for z, c in meas:
                    rows.append((t0, h, i, z, method, S[c].iloc[i] - seg[c].iloc[i]))
        for h, i in steps.items():
            for z, c in meas:
                rows.append((t0, h, i, z, "persistence", seg[c].iloc[0] - seg[c].iloc[i]))
    return pd.DataFrame(rows, columns=["origin", "horizon_h", "horizon_steps", "zone", "method", "error"]).dropna()
