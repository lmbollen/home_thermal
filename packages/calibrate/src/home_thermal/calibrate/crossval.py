"""Blocked cross-validation: the standard accuracy report.

The data after the warm-up is cut into blocks (default 3 days). Each block is held out in turn:
the model simulates straight through it, the fit ignores it, and fitted and prior parameters are
scored only there. The spread of fitted parameters across folds is an honest uncertainty (the
Laplace sd from a single fit is typically 2-5x too small). Optionally, forecasts from the filtered
state are scored in every held-out block.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from home_thermal.core import Network, defaults, parameter_set

from .estimate import forecast_errors, kalman
from .fit import FitSettings, fit, score


@dataclass
class CrossValidation:
    folds: pd.DataFrame          # one row per (fold, params, series): bias, rmse
    params: pd.DataFrame         # fitted parameters (+ derived H_house, H_party) per fold
    forecasts: pd.DataFrame | None
    free: list
    blocks: list

    def room_rmse(self) -> pd.DataFrame:
        """Held-out RMSE pooled over zone sensors, per fold: columns fitted / prior."""
        rooms = self.folds[self.folds.kind == "zone"]
        return rooms.groupby(["block", "params"]).rmse.apply(lambda r: float(np.sqrt((r ** 2).mean()))).unstack()

    def per_room(self) -> pd.DataFrame:
        rooms = self.folds[self.folds.kind == "zone"]
        return rooms.groupby(["series", "params"]).rmse.mean().unstack()

    def heat_source(self) -> pd.DataFrame:
        hp = self.folds[self.folds.kind == "heat source"]
        return hp.groupby(["series", "params"])[["bias", "rmse"]].mean().unstack()

    def forecast_rmse(self) -> pd.DataFrame | None:
        if self.forecasts is None or self.forecasts.empty:
            return None
        return (self.forecasts.groupby(["horizon_h", "method"]).error
                .apply(lambda e: float(np.sqrt((e ** 2).mean()))).unstack())

    def spread(self, cfg: dict) -> pd.DataFrame:
        """Prior vs fitted range across folds; at_bound = hit an optimiser bound in some fold."""
        ps = parameter_set(cfg)
        P = self.params
        out = pd.DataFrame({"prior": [ps[c].prior for c in self.free], "fold_mean": P[self.free].mean(),
                            "fold_min": P[self.free].min(), "fold_max": P[self.free].max()}, index=self.free)
        out["at_bound"] = [any(abs(P[c] - lim).min() < 1e-3 * max(1, abs(lim)) for lim in (ps[c].bounds or ()))
                           for c in self.free]
        return out

    def report(self, cfg: dict) -> str:
        r = self.room_rmse()
        lines = [f"{len(self.blocks)} folds",
                 "HELD-OUT zone temperature RMSE (K), pooled over zones, per fold:", r.round(2).to_string(),
                 f"mean over folds: fitted {r['fitted'].mean():.2f} K, prior {r['prior'].mean():.2f} K; "
                 f"fitted better in {(r['fitted'] < r['prior']).sum()}/{len(r)} folds",
                 "", "HELD-OUT RMSE per zone (K), mean over folds:", self.per_room().round(2).to_string()]
        hp = self.heat_source()
        if not hp.empty:
            lines += ["", "HELD-OUT heat source (W, hourly), mean over folds:", hp.round(0).to_string()]
        fr = self.forecast_rmse()
        if fr is not None:
            lines += ["", "HELD-OUT forecasts from the filtered state, RMSE (K):", fr.round(2).to_string()]
        P = self.params
        lines += ["", "PARAMETERS across folds:", self.spread(cfg).round(3).to_string(), "",
                  "DERIVED across folds: " + ", ".join(
                      f"{c} {P[c].mean():.0f} [{P[c].min():.0f}-{P[c].max():.0f}] W/K" for c in ("H_house", "H_party"))]
        return "\n".join(lines)


def make_blocks(index: pd.DatetimeIndex, warmup: pd.Timedelta, block: pd.Timedelta) -> list[tuple]:
    """Consecutive blocks after the warm-up; a tail shorter than half a block joins the last one."""
    t0 = index[0] + warmup
    step = index[1] - index[0]
    edges = list(pd.date_range(t0, index[-1], freq=block)) + [index[-1] + step]
    if len(edges) > 2 and (edges[-1] - edges[-2]) < block / 2:
        edges.pop(-2)
    if len(edges) < 3:
        raise ValueError(f"cross-validation needs at least two {block} blocks after the {warmup} warm-up")
    return list(zip(edges[:-1], edges[1:]))


def cross_validate(cfg: dict, df: pd.DataFrame, free: list | None = None, block="72h", warmup="48h",
                   settings: FitSettings | None = None, forecasts=True, forecast_every_h=6,
                   progress=print) -> CrossValidation:
    """Run blocked cross-validation. `progress` is called with one line per fold (None = silent).
    The warm-up is `settings.warmup` when settings are given, else `warmup`."""
    block = pd.Timedelta(block)
    settings = settings or FitSettings(warmup=warmup)
    warmup = settings.warmup
    blocks = make_blocks(df.index, warmup, block)
    prior = defaults(cfg)
    rows, params, fc = [], [], []
    for k, (a, b) in enumerate(blocks):
        test = (df.index >= a) & (df.index < b)
        res = fit(cfg, df, free=free, settings=settings, mask=~test)
        net = Network(cfg, res.params)
        params.append({**res.params, "H_house": net.H_house(), "H_party": net.H_parts["party"]})
        for label, par in (("fitted", res.params), ("prior", prior)):
            sc = score(cfg, par, df, test, warmup)
            for name, r in sc.iterrows():
                rows.append((k, f"{a:%d %b}-{b:%d %b}", label, name, r["kind"], r["bias"], r["rmse"]))
        if forecasts:
            origins = df.index[test & (df.index.hour % forecast_every_h == 0) & (df.index.minute == 0)]
            fc.append(forecast_errors(net, df, origins, filt=kalman(net, df)).assign(fold=k))
        if progress:
            progress(f"fold {k}: held out {a:%d %b %H:%M} .. {b:%d %b %H:%M} (cost {res.cost:.0f}, nfev {res.nfev})")
    return CrossValidation(pd.DataFrame(rows, columns=["fold", "block", "params", "series", "kind", "bias", "rmse"]),
                           pd.DataFrame(params), pd.concat(fc) if fc else None,
                           list(res.free), blocks)
