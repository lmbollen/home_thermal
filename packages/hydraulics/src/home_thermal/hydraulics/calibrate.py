"""Fit a circuit's $parameters to measurements (e.g. the heat pump's water flow).

    fit = fit_circuit(spec, ctxs, observed, predict=lambda res: res.flow["heat_pump"] * 1000)

Parameters are fitted in log space within their `bounds`, with a weak pull toward their starting
`value` (prior width `sigma`, default 0.5 in log space) so that parameters the data cannot see
stay put. Note that hydraulic data without an absolute reference cannot separate a flow sensor's
scale from the circuit's: a sensor that reads 20 % low gives kv values and pump flow 20 % low too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .spec import build_circuit


@dataclass
class CircuitFit:
    values: dict  # all parameter values (fitted + fixed)
    free: list
    rmse: float  # on the fitted rows, in the units of `observed`
    at_bound: list
    nfev: int


def predict_rows(spec: dict, values: dict, ctxs: list[dict], predict) -> np.ndarray:
    c = build_circuit(spec, values)
    out = np.empty(len(ctxs))
    for i, ctx in enumerate(ctxs):
        try:
            out[i] = predict(c.solve(ctx))
        except RuntimeError:
            out[i] = np.nan
    return out


def fit_circuit(
    spec: dict,
    ctxs: list[dict],
    observed,
    predict,
    free: list | None = None,
    sigma_obs: float = 1.0,
    max_nfev: int = 60,
    verbose: int = 0,
) -> CircuitFit:
    """Least-squares fit of the free parameters (default: all in spec['parameters'])."""
    params = spec.get("parameters") or {}
    free = list(free) if free is not None else list(params)
    for n in free:
        if n not in params or "bounds" not in params[n]:
            raise KeyError(f"parameter {n!r} needs a value and bounds in spec['parameters']")
    base = {n: float(v["value"]) for n, v in params.items()}
    x0 = np.log([base[n] for n in free])
    lo = np.log([params[n]["bounds"][0] for n in free])
    hi = np.log([params[n]["bounds"][1] for n in free])
    sig = np.array([float(params[n].get("sigma", 0.5)) for n in free])
    obs = np.asarray(observed, float)
    w = 1.0 / np.sqrt(len(obs))

    def values(x):
        return {**base, **dict(zip(free, np.exp(x)))}

    def fun(x):
        pred = predict_rows(spec, values(x), ctxs, predict)
        r = np.where(np.isnan(pred), 10.0, (pred - obs) / sigma_obs) * w
        return np.concatenate([r, (x - x0) / sig * 0.1])

    res = least_squares(
        fun,
        np.clip(x0, lo + 1e-9, hi - 1e-9),
        bounds=(lo, hi),
        x_scale=1.0,
        diff_step=1e-2,
        max_nfev=max_nfev,
        verbose=verbose,
    )
    vals = values(res.x)
    pred = predict_rows(spec, vals, ctxs, predict)
    at_bound = [n for n, x, a, b in zip(free, res.x, lo, hi) if min(x - a, b - x) < 1e-3]
    return CircuitFit(vals, free, float(np.sqrt(np.nanmean((pred - obs) ** 2))), at_bound, int(res.nfev))
