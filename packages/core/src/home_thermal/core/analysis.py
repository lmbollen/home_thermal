"""Where the heat goes: energy balance per source and loss path, and room-to-room coupling."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .network import BOUNDARIES, Network
from .physics import source_zone
from .simulate import simulate


def simulate_detailed(net: Network, df: pd.DataFrame, **kwargs) -> dict:
    """`simulate` with every heat input recorded per step and node (input to `energy_balance`).
    Keyword arguments go to `simulate` (e.g. controller=)."""
    X, Q, rec = simulate(net, df, record=True, **kwargs)
    return dict(rec, X=X.to_numpy(), Q=Q)


def energy_balance(net: Network, sim: dict, start=None) -> dict:
    """kWh per source, per loss path and between zones over the period (from `start`).

    sources: heat entering the house nodes (heating emitters, hot water into the tank, sun,
    background gain) plus, in group 'info', hot-water electricity and source heat (not extra heat).
    losses: heat leaving through boundaries, plus hot water drawn at the taps.
    stored: change in stored heat. sources (excluding 'info') - losses = stored, up to one step."""
    idx = sim["index"]
    if start is not None:
        start = pd.Timestamp(start)
        if idx.tz is not None and start.tzinfo is None:
            start = start.tz_localize(idx.tz)
    if start is not None and not (idx >= start).any():
        raise ValueError(f"energy_balance: no data after {start}")
    keep = np.ones(len(idx), bool) if start is None else np.asarray(idx >= start)
    X, Tb, Qe = sim["X"][keep], sim["Tb"][keep], sim["Qe"][keep]
    h = sim["dt"] / 3600 / 1000   # W*step -> kWh
    cfg = net.cfg
    hz = source_zone(cfg)[0]
    src = []
    for z in net.zones:
        e = Qe[:, net.iA[z]].sum() * h
        parts = ([f"radiator {z}"] if z in cfg["emitters"]["radiators"] else []) \
            + ([f"pipes {z}"] if z in cfg["emitters"]["pipes"] else []) \
            + ([f"heat source unit/pipes -> {z}"] if z == hz else [])
        if parts:
            src.append(("heating", " + ".join(parts), e))
    for z in net.ufh_zones:
        src.append(("heating", f"floor heating {z}", Qe[:, net.iS[z]].sum() * h))
    losses = []
    if net.iT is not None:
        tank_in, tap, extracted = (sim["tank"][keep] * h).sum(axis=0)
        cop = net.p["cop_dhw"]
        src.append(("hot water", "heat into tank", tank_in))
        src.append(("info", "hot water: electricity", tank_in / cop))
        src.append(("info", "hot water: heat taken from heating circuit", extracted))
        losses.append((cfg["dhw"]["zone"], "hot water drawn", "taps", tap))
    if np.any(sim.get("pipes", 0)):
        src.append(("info", "heat source: lost by the outdoor pipes", sim["pipes"][keep].sum() * h))
    src.append(("free", "sun through windows", sim["Qs"][keep].sum() * h))
    src.append(("free", "background gain", sim["Qi"][keep].sum() * h))
    if np.any(sim["Qx"]):
        src.append(("other", "extra heat (q_extra)", sim["Qx"][keep].sum() * h))
    sources = pd.DataFrame(src, columns=["group", "source", "kWh"])

    internal = []
    for L in net.links:
        if L.b is not None:
            f = L.g * (X[:, L.i] - Tb[:, L.b])
            losses.append((L.zone, L.category, BOUNDARIES[L.b], f.sum() * h))
        elif L.category in ("internal", "tank-loss"):
            f = L.g * (X[:, L.i] - X[:, L.j])   # + means first -> second
            pair = L.zone if L.category == "internal" else ("hot-water tank", L.zone)
            internal.append((pair[0], pair[1], f.mean(), np.abs(f).mean(), f.sum() * h))
    losses = pd.DataFrame(losses, columns=["zone", "path", "to", "kWh"])
    internal = pd.DataFrame(internal, columns=["from", "to", "mean_W", "mean_abs_W", "net_kWh"])
    stored = float((net.C * (X[-1] - X[0])).sum() / 3.6e6)
    return dict(sources=sources, losses=losses, internal=internal, stored_kWh=stored,
                hours=keep.sum() * sim["dt"] / 3600)


def coupling(net: Network) -> dict:
    """Steady-state room coupling (boundaries fixed).

    warming[i, j]: K rise of zone i per kW delivered into zone j.
    fate[i, j]:    share of heat delivered in zone j that leaves through zone i's envelope.
    Heat into a floor-heated zone is injected into its slab."""
    Ginv = np.linalg.inv(net.G)
    zones = net.zones
    inj = {z: (net.iS[z] if z in net.iS else net.iA[z]) for z in zones}
    gb = net.Gb.sum(axis=1)
    node_zone = np.empty(net.n, dtype=object)
    for z in zones:
        node_zone[net.iA[z]] = z
        node_zone[net.iM[z]] = z
    for z in net.ufh_zones:
        node_zone[net.iS[z]] = z
    if net.iT is not None:
        node_zone[net.iT] = net.cfg["dhw"]["zone"]
    warming = pd.DataFrame(0.0, index=zones, columns=zones)
    fate = pd.DataFrame(0.0, index=zones, columns=zones)
    for j in zones:
        dx = Ginv[:, inj[j]] * 1000.0
        for i in zones:
            warming.loc[i, j] = dx[net.iA[i]]
        out = gb * dx / 1000.0
        for k in range(net.n):
            fate.loc[node_zone[k], j] += out[k]
    return dict(warming=warming, fate=fate)
