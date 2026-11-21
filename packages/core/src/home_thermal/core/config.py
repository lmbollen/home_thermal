"""House configuration: load a YAML file, fill defaults and check it before any model is built.

The configuration is a plain dict (what YAML gives). `load_house` returns it normalised: every
optional section present, defaults filled in, and every key and cross-reference checked. Unknown
keys are errors (a typo would otherwise silently drop a section); free-text `note` keys are allowed
everywhere. Errors name the offending key. See docs/configuration.md for the schema.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

ENVELOPE_KINDS = ("wall", "roof", "window", "door", "floor", "party")
RADIATOR_MODELS = ("ntu", "catalogue")
BACKGROUND_MODES = ("constant", "meter")
PUMP_MODES = ("observed", "always", "follow")
PARAM_FIELDS = ("prior", "sigma", "bounds", "fixed")

DEFAULT_SOLAR_WEIGHTS = {"S": 1.0, "SW": 0.9, "SE": 0.9, "W": 0.6, "E": 0.6, "N": 0.2,
                         "NW": 0.3, "NE": 0.3, "H": 1.0}
DEFAULT_THERMAL_MASS = {  # ISO 13790 'medium' class per m2 floor area
    "air_furniture_factor": 5.0,
    "C_m_per_m2": 165000.0,
    "H_am_per_m2": 22.75,
}

# allowed keys per section (besides 'note')
TOP_KEYS = {"name", "zones", "envelope", "internal", "emitters", "heat_pump", "dhw", "solar",
            "background_gain", "boundaries", "thermal_mass", "parameters", "controls"}
ZONE_KEYS = {"area", "volume", "ach", "sensor", "occupied", "in_house_loss"}
ELEMENT_KEYS = {"kind", "area", "U", "solar", "ufh"}
EMITTER_KEYS = {"radiator_model", "valve_sensor_tau_h", "radiators", "floor_heating"}
RADIATOR_KEYS = {"q50", "n", "valve", "sensor_on_valve", "flow_lph"}
FLOOR_KEYS = {"area", "K_ws", "h_sa", "c", "pump"}
HEAT_PUMP_KEYS = {"zone", "dT_cond", "dT_evap", "standby_W", "running_min_W"}
DHW_KEYS = {"zone", "volume_l", "power", "on_threshold_W", "T_cold", "T_max"}
SOLAR_KEYS = {"proxy", "weights"}
BACKGROUND_KEYS = {"mode", "meter_column", "meter_heat_share"}
BOUNDARY_KEYS = {"T_ground", "T_neighbour"}
CONTROL_KEYS = {"thermostats", "hysteresis", "pumps"}


class ConfigError(ValueError):
    """The house configuration is invalid."""


def load_house(path: str | Path) -> dict:
    """Read a house YAML file and return the normalised configuration."""
    try:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"{path}: not valid YAML: {e}") from None
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return normalise(cfg)


def normalise(cfg: dict) -> dict:
    """Fill defaults and validate. Returns a new dict; the input is not modified."""
    c = copy.deepcopy(cfg)
    _keys(c, TOP_KEYS, "top level")
    if not isinstance(c.get("zones"), dict) or not c["zones"]:
        raise ConfigError("zones: at least one zone is required (a mapping of zone name -> settings)")
    c["name"] = c.get("name") or "house"
    for key in ("envelope", "emitters", "heat_pump", "solar", "background_gain", "boundaries",
                "thermal_mass", "parameters", "controls"):
        c[key] = _section(c, key)
    c["internal"] = c.get("internal") or []
    c["dhw"] = c.get("dhw") or None
    _check_zones(c)
    _check_envelope(c)
    _check_internal(c)
    _check_emitters(c)
    _check_heat_sources(c)
    _check_misc(c)
    _check_parameters(c)
    _check_controls(c)
    return c


def _check_zones(c):
    zones = c["zones"]
    for z, zc in zones.items():
        where = f"zones.{z}"
        if not isinstance(zc, dict):
            raise ConfigError(f"{where}: expected a mapping like {{area: 20, volume: 50}}")
        _keys(zc, ZONE_KEYS, where)
        _number(zc, "area", where, positive=True)
        _number(zc, "volume", where, positive=True)
        zc.setdefault("ach", 0.5)
        _number(zc, "ach", where, minimum=0)
        zc.setdefault("sensor", None)
        zc.setdefault("occupied", True)
        zc.setdefault("in_house_loss", True)
        _bool(zc, "occupied", where)
        _bool(zc, "in_house_loss", where)
        if zc["sensor"] is not None and not isinstance(zc["sensor"], str):
            raise ConfigError(f"{where}.sensor: a data column name, got {zc['sensor']!r}")
    if not any(zc["occupied"] for zc in zones.values()):
        raise ConfigError("zones: at least one zone must be occupied (receives the background gain)")
    if not any(zc["sensor"] for zc in zones.values()):
        raise ConfigError("zones: at least one zone needs a 'sensor' column to calibrate against")


def _check_envelope(c):
    zones, weights = c["zones"], None
    c["solar"].setdefault("proxy", "PV_W")
    weights = c["solar"]["weights"] = {**DEFAULT_SOLAR_WEIGHTS, **(c["solar"].get("weights") or {})}
    for z in list(c["envelope"]):
        _zone(z, zones, "envelope")
        elements = c["envelope"][z] = c["envelope"][z] or []
        if not isinstance(elements, list):
            raise ConfigError(f"envelope.{z}: expected a list of elements")
        for i, el in enumerate(elements):
            where = f"envelope.{z}[{i}]"
            if not isinstance(el, dict):
                raise ConfigError(f"{where}: expected a mapping like {{kind: wall, area: 10, U: 0.5}}")
            _keys(el, ELEMENT_KEYS, where)
            if el.get("kind") not in ENVELOPE_KINDS:
                raise ConfigError(f"{where}.kind: must be one of {ENVELOPE_KINDS}, got {el.get('kind')!r}")
            _number(el, "area", where, positive=True)
            _number(el, "U", where, positive=True)
            if "solar" in el:
                if el["kind"] not in ("window", "door"):
                    raise ConfigError(f"{where}.solar: only windows and doors receive solar gain")
                if el["solar"] not in weights:
                    raise ConfigError(f"{where}.solar: unknown orientation {el['solar']!r} "
                                      f"(known: {sorted(weights)}; add it under solar.weights)")
            if "ufh" in el:
                _bool(el, "ufh", where)
            if el.get("ufh") and el["kind"] != "floor":
                raise ConfigError(f"{where}.ufh: only floor elements can be heated floors")


def _check_internal(c):
    zones = c["zones"]
    for i, link in enumerate(c["internal"]):
        where = f"internal[{i}]"
        if not (isinstance(link, (list, tuple)) and len(link) in (3, 4)):
            raise ConfigError(f"{where}: expected [zone, zone, UA_W_per_K] or [zone, zone, UA, air]")
        _zone(link[0], zones, where)
        _zone(link[1], zones, where)
        if link[0] == link[1]:
            raise ConfigError(f"{where}: links a zone to itself")
        if not _is_number(link[2]) or link[2] < 0:
            raise ConfigError(f"{where}: UA must be a number >= 0, got {link[2]!r}")
        if len(link) == 4 and link[3] not in ("air", "solid"):
            raise ConfigError(f"{where}: 4th field must be 'air' or 'solid', got {link[3]!r}")


def _check_emitters(c):
    zones, em = c["zones"], c["emitters"]
    _keys(em, EMITTER_KEYS, "emitters")
    em.setdefault("radiator_model", "ntu")
    em.setdefault("valve_sensor_tau_h", 0.5)
    _number(em, "valve_sensor_tau_h", "emitters", positive=True)
    em["radiators"] = em.get("radiators") or {}
    em["floor_heating"] = em.get("floor_heating") or {}
    if em["radiator_model"] not in RADIATOR_MODELS:
        raise ConfigError(f"emitters.radiator_model: must be one of {RADIATOR_MODELS}")
    for z, r in em["radiators"].items():
        where = f"emitters.radiators.{z}"
        _zone(z, zones, where)
        r = _mapping(r, where)
        _keys(r, RADIATOR_KEYS, where)
        _number(r, "q50", where, positive=True)
        r.setdefault("n", 1.3)
        _number(r, "n", where, positive=True)
        if "flow_lph" in r:
            _number(r, "flow_lph", where, positive=True)
        if "valve" not in r:
            raise ConfigError(f"{where}.valve: required: a data column with the valve %, or a fixed % "
                              "(e.g. 100 for a radiator without valve data)")
        v = r["valve"]
        if not (isinstance(v, str) or (_is_number(v) and 0 <= v <= 100)):
            raise ConfigError(f"{where}.valve: a column name or a number 0-100, got {v!r}")
        r.setdefault("sensor_on_valve", False)
        _bool(r, "sensor_on_valve", where)
        if r["sensor_on_valve"] and not zones[z]["sensor"]:
            raise ConfigError(f"{where}.sensor_on_valve: zone {z} has no sensor")
    for z, u in em["floor_heating"].items():
        where = f"emitters.floor_heating.{z}"
        _zone(z, zones, where)
        u = _mapping(u, where)
        _keys(u, FLOOR_KEYS, where)
        for k in ("area", "K_ws", "h_sa", "c"):
            _number(u, k, where, positive=True)
        u.setdefault("pump", "always")
        if not isinstance(u["pump"], str):
            raise ConfigError(f"{where}.pump: a data column name or 'always'")
        if u["pump"].lower() == "always":
            u["pump"] = "always"


def _check_heat_sources(c):
    zones, hp = c["zones"], c["heat_pump"]
    _keys(hp, HEAT_PUMP_KEYS, "heat_pump")
    if hp.get("zone") is not None:
        _zone(hp["zone"], zones, "heat_pump.zone")
    hp.setdefault("zone", None)
    for k, v in (("dT_cond", 3.0), ("dT_evap", 6.0), ("standby_W", 7.0), ("running_min_W", 50.0)):
        hp.setdefault(k, v)
        _number(hp, k, "heat_pump", minimum=0)
    if c["dhw"]:
        d = c["dhw"]
        _keys(d, DHW_KEYS, "dhw")
        _zone(d.get("zone"), zones, "dhw.zone")
        _number(d, "volume_l", "dhw", positive=True)
        if not isinstance(d.get("power"), str):
            raise ConfigError("dhw.power: required: the data column with the hot-water heat pump's electricity (W)")
        for k, v in (("on_threshold_W", 5.0), ("T_cold", 10.0), ("T_max", 60.0)):
            d.setdefault(k, v)
            _number(d, k, "dhw")


def _check_misc(c):
    _keys(c["solar"], SOLAR_KEYS, "solar")
    bg = c["background_gain"] = {"mode": "constant", "meter_column": "P_app_W", "meter_heat_share": 0.8,
                                 **c["background_gain"]}
    _keys(bg, BACKGROUND_KEYS, "background_gain")
    if bg["mode"] not in BACKGROUND_MODES:
        raise ConfigError(f"background_gain.mode: must be one of {BACKGROUND_MODES}")
    _keys(c["boundaries"], BOUNDARY_KEYS, "boundaries")
    for k in c["boundaries"]:
        _number(c["boundaries"], k, "boundaries")
    c["thermal_mass"] = {**DEFAULT_THERMAL_MASS, **c["thermal_mass"]}
    _keys(c["thermal_mass"], set(DEFAULT_THERMAL_MASS), "thermal_mass")
    for k in DEFAULT_THERMAL_MASS:
        _number(c["thermal_mass"], k, "thermal_mass", positive=True)


def _check_parameters(c):
    for name, ov in c["parameters"].items():
        where = f"parameters.{name}"
        if not isinstance(ov, dict):
            raise ConfigError(f"{where}: expected a mapping like {{prior: 1.0, fixed: true}}")
        _keys(ov, set(PARAM_FIELDS), where, allow_note=False)
        for k in ("prior", "sigma"):
            if k in ov:
                _number(ov, k, where, positive=(k == "sigma"))
        if "fixed" in ov:
            _bool(ov, "fixed", where)
        b = ov.get("bounds")
        if b is not None and not (isinstance(b, list) and len(b) == 2 and all(map(_is_number, b)) and b[0] < b[1]):
            raise ConfigError(f"{where}.bounds: expected [low, high] with low < high, or null")
    # names are checked against the house's parameter set (needs the rest of the config)
    from .params import parameter_set
    try:
        parameter_set(c)
    except (KeyError, ValueError) as e:
        raise ConfigError(e.args[0]) from None


def _check_controls(c):
    ctl, zones = c["controls"], c["zones"]
    _keys(ctl, CONTROL_KEYS, "controls")
    th = ctl["thermostats"] = ctl.get("thermostats") or {}
    for z, t in th.items():
        where = f"controls.thermostats.{z}"
        _zone(z, zones, where)
        t = th[z] = t or {}
        _keys(t, {"valve", "setpoint", "group"}, where)
        if not zones[z]["sensor"]:
            raise ConfigError(f"{where}: zone {z} has no sensor for the thermostat to read")
        t.setdefault("valve", f"V_{z}")
        t.setdefault("setpoint", f"SP_{z}")
        t.setdefault("group", z)
    for name, h in (ctl.setdefault("hysteresis", None) or {}).items():
        where = f"controls.hysteresis.{name}"
        h = _mapping(h, where)
        _keys(h, {"zone", "setpoint", "below", "above"}, where)
        _zone(h.get("zone"), zones, where + ".zone")
        if not isinstance(h.get("setpoint"), str):
            raise ConfigError(f"{where}.setpoint: required: the setpoint data column")
        h.setdefault("below", 0.3)
        h.setdefault("above", 0.3)
    ctl["hysteresis"] = ctl["hysteresis"] or {}
    for name, p in (ctl.setdefault("pumps", None) or {}).items():
        where = f"controls.pumps.{name}"
        p = _mapping(p, where)
        _keys(p, {"mode", "follow"}, where)
        p.setdefault("mode", "observed")
        if p["mode"] not in PUMP_MODES:
            raise ConfigError(f"{where}.mode: must be one of {PUMP_MODES}")
        if p["mode"] == "follow" and not isinstance(p.get("follow"), str):
            raise ConfigError(f"{where}.follow: required for mode 'follow': the valve column to follow")
    ctl["pumps"] = ctl["pumps"] or {}


# ---------------------------------------------------------------------------- helpers

def _section(c: dict, key: str) -> dict:
    v = c.get(key)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ConfigError(f"{key}: expected a mapping, got {type(v).__name__}")
    return v


def _keys(d: dict, allowed: set, where: str, allow_note: bool = True) -> None:
    unknown = set(d) - allowed - ({"note"} if allow_note else set())
    if unknown:
        hints = []
        for u in sorted(map(str, unknown)):
            close = [a for a in allowed if a.lower() == u.lower() or _near(a, u)]
            hints.append(f"{u!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ConfigError(f"{where}: unknown key {', '.join(hints)}; allowed: {', '.join(sorted(allowed))}")


def _near(a: str, b: str) -> bool:
    """One edit apart (typo)."""
    a, b = a.lower(), b.lower()
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, t = (a, b) if len(a) < len(b) else (b, a)
    return any(t[:i] + t[i + 1:] == s for i in range(len(t)))


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _mapping(v, where: str) -> dict:
    if not isinstance(v, dict):
        raise ConfigError(f"{where}: expected a mapping, got {v!r}")
    return v


def _bool(d: dict, key: str, where: str) -> None:
    if not isinstance(d.get(key), bool):
        raise ConfigError(f"{where}.{key}: must be true or false, got {d.get(key)!r}")


def _number(d: dict, key: str, where: str, positive: bool = False, minimum: float | None = None) -> None:
    if key not in d:
        found = [k for k in d if str(k).lower() == key.lower()]
        hint = f" (found {found[0]!r}: keys are case-sensitive)" if found else ""
        raise ConfigError(f"{where}.{key}: missing{hint}")
    v = d[key]
    if not _is_number(v):
        raise ConfigError(f"{where}.{key}: must be a number, got {v!r}")
    if positive and v <= 0:
        raise ConfigError(f"{where}.{key}: must be > 0, got {v!r}")
    if minimum is not None and v < minimum:
        raise ConfigError(f"{where}.{key}: must be >= {minimum}, got {v!r}")


def _zone(z, zones: dict, where: str) -> None:
    if z not in zones:
        raise ConfigError(f"{where}: unknown zone {z!r} (zones: {', '.join(zones)})")
