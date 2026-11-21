"""Model parameters: what each one means, its prior, and which houses it applies to.

Most parameters are multipliers ('log' scale) on a physical prior that comes from the house
configuration (U-values, areas, catalogue outputs). A multiplier of 1 means "the config was right".
The rest are physical quantities ('lin' scale) such as the ground temperature.

`parameter_set(cfg)` returns only the parameters that exist for this house (no radiator
parameters without radiators, ...), with the house's own overrides from `parameters:` applied.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

LOG_BOUNDS = (0.3, 3.0)


@dataclass(frozen=True)
class Param:
    prior: float
    sigma: float          # prior standard deviation (in log space for scale='log')
    scale: str            # 'log' (multiplier) or 'lin'
    bounds: tuple | None  # hard optimiser bounds; None = unbounded
    fixed: bool = False   # held at the prior unless the config frees it
    doc: str = ""


def _m(sigma, doc, bounds=LOG_BOUNDS, fixed=False, prior=1.0):
    return Param(prior, sigma, "log", bounds, fixed, doc)


def _lin(prior, sigma, bounds, doc, fixed=False):
    return Param(prior, sigma, "lin", bounds, fixed, doc)


# Generic defaults. 'fixed' marks parameters that room temperatures alone rarely identify; fitting
# them tends to let them compensate for other errors.
GENERIC: dict[str, Param] = {
    "m_env": _m(0.25, "opaque envelope (walls, roofs) UA multiplier"),
    "m_win": _m(0.15, "windows and doors UA multiplier (label U-values are fairly reliable)", fixed=True),
    "m_vent": _m(0.50, "infiltration / natural ventilation multiplier"),
    "m_party": _m(0.50, "party wall to the neighbour UA multiplier", bounds=(0.5, 1.5)),
    "m_floor": _m(0.40, "floor to ground UA multiplier", bounds=(0.5, 2.0)),
    "m_int": _m(0.40, "internal walls/floors between zones ('solid' links)"),
    "m_air": _m(0.70, "door/stair air exchange ('air' links); doors are not measured", bounds=(0.3, 5.0),
                fixed=True),
    "m_Cair": _m(0.50, "air + furniture heat capacity multiplier", fixed=True),
    "m_Cm": _m(0.50, "structural mass heat capacity multiplier"),
    "m_Ham": _m(0.50, "air <-> mass coupling multiplier"),
    "g_sol": Param(6.0, 0.50, "log", (0.05, 20.0), False,
                   "W solar gain through all windows per W of the solar proxy (e.g. PV output)"),
    "Q_int": _lin(200.0, 120.0, (0.0, 800.0),
                  "background gain (W): average heat from people/appliances, not a model of them"),
    "T_ground": _lin(11.0, 1.0, (8.0, 14.0), "ground / crawl space temperature under the floor (C)"),
    "T_neighbour": _lin(16.0, 2.0, (10.0, 22.0), "temperature behind party walls (C)"),
    # floor heating
    "m_ufh": _m(0.40, "water -> slab conductance multiplier"),
    # radiators
    "m_rad": _m(0.40, "radiator output vs catalogue q50"),
    "v0": _lin(10.0, 8.0, (0.0, 40.0), "valve % below which no water flows (ntu)"),
    "m_flow": _m(0.70, "radiator water flow at 100 % vs design (ntu); trades off with m_rad",
                 bounds=(0.1, 10.0), fixed=True),
    "kappa_q": _lin(0.3, 0.3, (0.0, 3.0), "valve-mounted sensor warming, K per 100 W radiator output (ntu)"),
    "gamma_v": Param(1.5, 0.5, "log", (0.3, 5.0), False, "valve characteristic exponent (catalogue)"),
    "kappa_s": _lin(0.10, 0.10, (0.0, 0.6), "valve-mounted sensor share of radiator excess temperature (catalogue)"),
    # heat pump
    "eta_hp": Param(0.52, 0.15, "log", (0.3, 0.7), False, "heat pump COP as fraction of Carnot"),
    "P_aux_hp": _lin(50.0, 30.0, (0.0, 200.0), "heat pump auxiliary power while running (W)"),
    "k_meter": Param(1.0, 0.15, "log", (0.6, 1.3), False, "heat meter scale error"),
    "dT_off": _lin(0.0, 0.15, (-1.5, 1.5), "flow/return sensor offset in the heat meter's own dT (K); "
                   "set the prior to your measured standstill offset", fixed=True),
    "UA_hp": _lin(10.0, 10.0, (0.0, 60.0), "heat pump unit + pipework losses into its zone (W/K)"),
    # domestic hot water tank
    "cop_dhw": Param(3.5, 0.2, "log", (2.0, 5.0), True, "hot-water heat pump COP"),
    "UA_tank": Param(1.5, 0.4, "log", (0.5, 5.0), True, "hot-water tank loss (W/K)"),
    "tap_W": _lin(120.0, 80.0, (0.0, 400.0), "average hot-water tap draw (W)", fixed=True),
}
# residual flow through a floor-heating circuit while its pump is off: one per pump column
F_OFF = _lin(0.05, 0.2, (0.0, 1.0), "residual flow fraction with the pump off")


# parameters that only matter when the data has heat-pump measurements (HP_heat_W / HP_elec_W)
HEAT_PUMP_PARAMS = ("eta_hp", "P_aux_hp", "k_meter", "dT_off")


def parameter_set(cfg: dict) -> dict[str, Param]:
    """Parameters that exist for this (normalised) house configuration, overrides applied.

    A parameter exists only if something in the house uses it (no m_rad without radiators, no
    T_neighbour without party walls). `boundaries:` in the config sets the priors of T_ground and
    T_neighbour; `parameters:` overrides anything."""
    em, elements = cfg["emitters"], [el for els in cfg["envelope"].values() for el in els]
    kinds = {el["kind"] for el in elements}
    link_kinds = {link[3] if len(link) == 4 else "solid" for link in cfg["internal"]}
    out = {n: GENERIC[n] for n in ("m_env", "m_win", "m_vent", "m_Cair", "m_Cm", "m_Ham", "Q_int")}
    if "floor" in kinds:
        out["m_floor"] = GENERIC["m_floor"]
        out["T_ground"] = GENERIC["T_ground"]
    if "party" in kinds:
        out["m_party"] = GENERIC["m_party"]
        out["T_neighbour"] = GENERIC["T_neighbour"]
    if "solid" in link_kinds:
        out["m_int"] = GENERIC["m_int"]
    if "air" in link_kinds:
        out["m_air"] = GENERIC["m_air"]
    if any("solar" in el for el in elements):
        out["g_sol"] = GENERIC["g_sol"]
    if em["floor_heating"]:
        out["m_ufh"] = GENERIC["m_ufh"]
        for u in em["floor_heating"].values():
            if u["pump"] != "always":
                out[f"f_off_{u['pump']}"] = F_OFF
    if em["radiators"]:
        out["m_rad"] = GENERIC["m_rad"]
        on_valve = any(r["sensor_on_valve"] for r in em["radiators"].values())
        model_params = ("v0", "m_flow", "kappa_q") if em["radiator_model"] == "ntu" else ("gamma_v", "kappa_s")
        out.update({n: GENERIC[n] for n in model_params[:-1]})
        if on_valve:
            out[model_params[-1]] = GENERIC[model_params[-1]]
    out.update({n: GENERIC[n] for n in HEAT_PUMP_PARAMS})
    if cfg["heat_pump"].get("zone"):
        out["UA_hp"] = GENERIC["UA_hp"]
    if cfg["dhw"]:
        out.update({n: GENERIC[n] for n in ("cop_dhw", "UA_tank", "tap_W")})

    for name, value in cfg["boundaries"].items():   # prior, with the default bounds moved along
        if name in out:
            half = (out[name].bounds[1] - out[name].bounds[0]) / 2
            out[name] = replace(out[name], prior=float(value), bounds=(value - half, value + half))
    for name, ov in cfg["parameters"].items():
        if name not in out:
            raise KeyError(f"parameters.{name}: not a parameter of this house (it has: {', '.join(out)})")
        ov = {k: v for k, v in ov.items() if k != "note"}
        if ov.get("bounds") is not None:
            ov["bounds"] = tuple(ov["bounds"])
        out[name] = replace(out[name], **ov)
    for name, par in out.items():
        lo, hi = par.bounds or (-float("inf"), float("inf"))
        if par.scale == "log" and (par.prior <= 0 or hi <= 0):
            raise ValueError(f"parameters.{name}: prior and bounds must be > 0 (it is a log-scale parameter)")
        if not lo <= par.prior <= hi:
            raise ValueError(f"parameters.{name}: prior {par.prior} is outside its bounds {par.bounds}")
    return out


def check_names(cfg: dict, p: dict) -> None:
    """Raise KeyError for parameter names that are not parameters of this house (typos)."""
    unknown = sorted(set(p) - set(parameter_set(cfg)))
    if unknown:
        raise KeyError(f"not parameters of this house: {unknown} (see `home-thermal check`)")


def defaults(cfg: dict) -> dict[str, float]:
    """Every parameter at its prior."""
    return {k: v.prior for k, v in parameter_set(cfg).items()}


def default_free(cfg: dict) -> list[str]:
    """Parameters calibrated by default for this house."""
    return [k for k, v in parameter_set(cfg).items() if not v.fixed]


def complete(cfg: dict, p: dict, strict: bool = True) -> dict[str, float]:
    """Parameters for this house: priors, overwritten by p. Unknown names in p raise KeyError,
    or are dropped with strict=False (e.g. loading a file saved for a different config)."""
    if strict:
        check_names(cfg, p)
    base = defaults(cfg)
    return {**base, **{k: float(v) for k, v in p.items() if k in base}}
