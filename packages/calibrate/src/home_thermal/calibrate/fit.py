"""Grey-box calibration: fit parameters to measured room temperatures and heat-pump data,
regularised by the priors (maximum a posteriori with Gaussian priors).

Data terms (hourly means after the warm-up):
- every zone sensor,
- heat-meter output if the data has HP_heat_W (weak: meters are often biased),
- heat-pump electricity if the data has HP_elec_W and T_flow (strong),
- boiler gas use if the house has a `boiler` and the data its gas column (strong; hours in which
  the boiler heated tap water are left out).
Model errors are strongly autocorrelated, so the data term counts roughly one independent sample
per `autocorr_h` hours; otherwise the priors would be ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from home_thermal.core import (
    DATA_TERM_PARAMS,
    Boiler,
    HeatPump,
    Network,
    check_names,
    check_sensors,
    default_free,
    defaults,
    dhw_steps,
    gas_power,
    parameter_set,
    prepare,
    sensor_temps,
    simulate,
)


@dataclass
class FitSettings:
    sigma_T: float = 0.5        # K, hourly room temperature model error (incl. sensor placement)
    sigma_Q: float = 800.0      # W, hourly heat-meter error (weak constraint)
    sigma_P: float = 100.0      # W, hourly heat-pump electricity error (strong anchor)
    sigma_G: float = 150.0      # W, hourly boiler gas power error (strong anchor)
    autocorr_h: float = 6.0
    warmup: pd.Timedelta = field(default_factory=lambda: pd.Timedelta("24h"))
    max_nfev: int = 300

    def __post_init__(self):
        self.warmup = pd.Timedelta(self.warmup)


@dataclass
class FitResult:
    params: dict                 # all parameters (fitted + fixed)
    free: list
    sd: dict                     # Laplace approximation; typically 2-5x too small (use cross-validation)
    corr: np.ndarray | None
    cost: float
    nfev: int
    at_bound: list               # free parameters that ended on an optimiser bound


def meter_heat_true(df: pd.DataFrame, p: dict) -> pd.Series:
    """Heat-meter output corrected for its flow/return sensor offset dT_off and scale k_meter.

    Meters that compute Q = c * dT_raw report too much heat when dT_raw contains a sensor offset:
    Q_true = k_meter * (Q_rep - dT_off * Q_rep/dT_raw). Q_rep/dT_raw comes from the data source as
    HP_heat_per_K (absent = no offset correction)."""
    per_k = df["HP_heat_per_K"].fillna(0) if "HP_heat_per_K" in df else 0.0
    return p["k_meter"] * (df["HP_heat_W"] - p["dT_off"] * per_k).clip(lower=0)


def data_terms(cfg: dict, df: pd.DataFrame) -> set[str]:
    """Heat-source data terms this house and data support: 'heat meter', 'electricity', 'gas'."""
    terms = set()
    if "HP_heat_W" in df:
        terms.add("heat meter")
    if cfg["boiler"] is not None:
        if cfg["boiler"]["gas_column"] in df:
            terms.add("gas")
    elif {"HP_elec_W", "T_flow"} <= set(df.columns):
        terms.add("electricity")
    return terms


def _keep(df, warmup, mask):
    keep = df.index >= df.index[0] + pd.Timedelta(warmup)
    if mask is not None:
        keep = keep & np.asarray(mask)
    if not keep.any():
        raise ValueError(f"no data left after the {pd.Timedelta(warmup)} warm-up (and mask)")
    return keep


def _errors(cfg, net, X, Q, df, keep, hourly_rooms: bool) -> dict:
    """Model minus measurement per series: {name: (kind, errors)}; kind 'zone' or 'heat source'."""
    out = {}
    S = sensor_temps(net, X, df)
    for z, zc in cfg["zones"].items():
        if zc["sensor"]:
            e = (S[zc["sensor"]] - df[zc["sensor"]])[keep]
            out[z] = ("zone", (e.resample("1h").mean() if hourly_rooms else e).dropna())
    terms = data_terms(cfg, df)
    full_hours = pd.Series(keep, index=df.index).resample("1h").min().astype(bool)
    if "heat meter" in terms:
        h = (Q - meter_heat_true(df, net.p))[keep].resample("1h").mean().dropna()
        out["HP heat (W, hourly)"] = ("heat source", h)
    if "electricity" in terms:
        e = HeatPump.from_config(cfg, net.p).electric_hourly(Q, df) - df["HP_elec_W"].resample("1h").mean()
        out["HP electricity (W, hourly)"] = ("heat source", e[full_hours].dropna())
    if "gas" in terms:
        no_dhw = ~dhw_steps(cfg, df).resample("1h").max().astype(bool)
        pred = pd.Series(Boiler.from_config(cfg, net.p).fuel(Q), index=df.index).resample("1h").mean()
        e = pred - gas_power(cfg, df).resample("1h").mean()
        out["gas (W, hourly)"] = ("heat source", e[full_hours & no_dhw].dropna())
    return out


def residuals(cfg: dict, p: dict, df: pd.DataFrame, settings: FitSettings, mask=None, use_heat=True):
    """Weighted residual vector (data part only), plus the simulated states and heat."""
    net = Network(cfg, p)
    X, Q = simulate(net, df)
    w = 1.0 / np.sqrt(settings.autocorr_h)
    sigma = {"HP heat (W, hourly)": settings.sigma_Q, "HP electricity (W, hourly)": settings.sigma_P,
             "gas (W, hourly)": settings.sigma_G}
    errors = _errors(cfg, net, X, Q, df, _keep(df, settings.warmup, mask), True)
    r = [w * e.to_numpy() / sigma.get(name, settings.sigma_T)
         for name, (kind, e) in errors.items() if use_heat or kind == "zone"]
    return np.concatenate(r), X, Q


def _theta_bounds(ps, free):
    """Optimiser bounds in theta space (log for multipliers); None = unbounded."""
    lo, hi = [], []
    for n in free:
        b = ps[n].bounds
        if b is None:
            lo.append(-np.inf)
            hi.append(np.inf)
        elif ps[n].scale == "log":
            lo.append(np.log(b[0]) if b[0] > 0 else -np.inf)
            hi.append(np.log(b[1]))
        else:
            lo.append(b[0])
            hi.append(b[1])
    return np.array(lo), np.array(hi)


def fit(cfg: dict, df: pd.DataFrame, free: list | None = None, base: dict | None = None,
        settings: FitSettings | None = None, mask=None, use_heat=True, verbose=0) -> FitResult:
    """Fit the `free` parameters (default: `default_free(cfg)`); others stay at `base`/prior.
    mask: boolean array, True = step used in the fit (e.g. to hold out a test block).
    Heat-source parameters whose data term is missing (e.g. eta_boiler without gas data) are
    dropped from `free`: nothing in the data constrains them."""
    settings = settings or FitSettings()
    df = prepare(df)
    check_sensors(cfg, df)
    _keep(df, settings.warmup, mask)
    ps = parameter_set(cfg)
    free = list(free) if free is not None else default_free(cfg)
    check_names(cfg, dict.fromkeys(free))
    check_names(cfg, base or {})
    terms = data_terms(cfg, df) if use_heat else set()
    free = [n for n in free if DATA_TERM_PARAMS.get(n, "always") in terms | {"always"}]
    base = {**defaults(cfg), **(base or {})}
    if not free:
        return FitResult(base, [], {}, None, 0.0, 0, [])

    def to_theta(vals):
        return np.array([np.log(vals[n]) if ps[n].scale == "log" else vals[n] for n in free], float)

    def from_theta(theta):
        p = dict(base)
        for n, v in zip(free, theta):
            p[n] = float(np.exp(v)) if ps[n].scale == "log" else float(v)
        return p

    prior_theta = to_theta({n: ps[n].prior for n in free})
    prior_sigma = np.array([ps[n].sigma for n in free])
    lo, hi = _theta_bounds(ps, free)

    def fun(theta):
        r, _, _ = residuals(cfg, from_theta(theta), df, settings, mask, use_heat)
        return np.concatenate([r, (theta - prior_theta) / prior_sigma])

    x0 = np.clip(to_theta(base), lo + 1e-6, hi - 1e-6)
    res = least_squares(fun, x0, bounds=(lo, hi), method="trf", x_scale="jac",
                        diff_step=1e-3, verbose=verbose, max_nfev=settings.max_nfev)
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac)
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
    except np.linalg.LinAlgError:
        sd, corr = np.full(len(free), np.nan), None
    at_bound = [n for n, v, a, b in zip(free, res.x, lo, hi)
                if min(abs(v - a), abs(v - b)) < 1e-3 * max(1.0, abs(v))]
    return FitResult(from_theta(res.x), free, dict(zip(free, sd)), corr, float(res.cost), int(res.nfev), at_bound)


def score(cfg: dict, p: dict, df: pd.DataFrame, mask=None, warmup="24h") -> pd.DataFrame:
    """Bias and RMSE per zone sensor (K, at the data resolution) and, if measured, for the heat
    source: heat meter, heat-pump electricity, boiler gas (W, hourly), over the steps in mask
    after the warm-up.
    Columns: kind ('zone' | 'heat source'), bias, rmse."""
    df = prepare(df)
    check_sensors(cfg, df)
    net = Network(cfg, p)
    X, Q = simulate(net, df)
    errors = _errors(cfg, net, X, Q, df, _keep(df, warmup, mask), False)
    rows = {name: {"kind": kind, "bias": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean()))}
            for name, (kind, e) in errors.items()}
    return pd.DataFrame(rows).T.astype({"bias": float, "rmse": float})
