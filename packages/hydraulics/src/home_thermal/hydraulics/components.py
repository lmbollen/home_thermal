"""Hydraulic components. Units used throughout:

    flow q          m3/h (positive in the component's direction a -> b)
    pressure dp     kPa (1 m of water head = 9.78 kPa; 1 bar = 100 kPa)
    temperature     C
    kv              m3/h at 1 bar: q = kv * sqrt(dp / 100 kPa)

Most components are *elements* with a pressure drop dp(q) and an outlet temperature; elements in
series form one branch of the circuit (`Circuit.add`). An `OverflowValve` is a branch of its own,
because its flow is a function of the pressure across it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

CP = 4186.0  # J/kgK
RHO = 993.0  # kg/m3 near 35 C
G = 9.81
KPA_PER_M = RHO * G / 1000.0


def w_per_k(q: float) -> float:
    """Heat capacity flow (W/K) of q m3/h of water."""
    return abs(q) * RHO / 3600.0 * CP


def kv_dp(q: float, kv: float) -> float:
    """Pressure drop (kPa) of q m3/h through kv (signed, symmetric)."""
    return 100.0 * q * abs(q) / (kv * kv)


class Element:
    """A two-port element in a series branch."""

    name: str = ""

    def dp(self, q: float, ctx: dict) -> float:  # pressure drop a -> b, kPa
        return 0.0

    def outlet(self, t_in: float, q: float, ctx: dict) -> float:
        return t_in

    def heat(self, t_in: float, q: float, ctx: dict) -> float:
        """Heat (W) the element adds to the water (negative: takes out)."""
        return w_per_k(q) * (self.outlet(t_in, q, ctx) - t_in)


# ------------------------------------------------------------------ resistances


@dataclass
class Kv(Element):
    """A fixed resistance given by its kv (valve fully open, heat exchanger, fittings)."""

    kv: float
    name: str = "kv"

    def dp(self, q, ctx):
        return kv_dp(q, self.kv)


@dataclass
class Pipe(Element):
    """Straight pipe with fittings: Darcy-Weisbach with Blasius / laminar friction, plus the sum
    of local loss coefficients `zeta` (bends, tees). Inner diameter in m, length in m."""

    length: float
    diameter: float
    zeta: float = 0.0
    nu: float = 0.7e-6  # kinematic viscosity at ~35 C, m2/s
    name: str = "pipe"

    def dp(self, q, ctx):
        area = math.pi * self.diameter**2 / 4
        v = q / 3600.0 / area
        re = max(abs(v) * self.diameter / self.nu, 1e-9)
        lam = 64 / re if re < 2300 else 0.3164 * re**-0.25
        return (lam * self.length / self.diameter + self.zeta) * RHO * v * abs(v) / 2 / 1000.0


@dataclass
class Valve(Element):
    """Control valve: kv = kv_max x characteristic(opening), never below kv_min (leakage).
    `opening` is a fraction 0..1, or a callable(ctx) -> fraction (e.g. a data column).
    characteristic: 'linear', 'equal' (equal percentage, rangeability 30) or 'quick'."""

    kv_max: float
    opening: float | Callable = 1.0
    characteristic: str = "linear"
    kv_min: float = 1e-3
    name: str = "valve"

    def position(self, ctx) -> float:
        u = self.opening(ctx) if callable(self.opening) else self.opening
        return min(max(float(u), 0.0), 1.0)

    def kv(self, ctx) -> float:
        u = self.position(ctx)
        f = {"linear": u, "quick": math.sqrt(u), "equal": 30.0 ** (u - 1) if u > 0 else 0.0}[self.characteristic]
        return max(self.kv_max * f, self.kv_min)

    def dp(self, q, ctx):
        return kv_dp(q, self.kv(ctx))


# ------------------------------------------------------------------ pumps


@dataclass
class PumpCurve:
    """Maximum-speed curve of a pump: head H(q) = H0 (1 - (q/q0)^2) in m; H0 shut-off head (m),
    q0 flow at zero head (m3/h). A speed fraction s scales it by the affinity laws:
    H = s^2 H0 - (H0/q0^2) q^2."""

    H0: float
    q0: float
    name: str = ""

    def head(self, q: float, speed: float = 1.0) -> float:
        return speed**2 * self.H0 - self.H0 / self.q0**2 * q * abs(q)


@dataclass
class Pump(Element):
    """Circulation pump. mode:
      'speed'  constant speed, `setting` = speed fraction 0..1 (Wilo I/II/III ~ 0.6/0.8/1.0)
      'dp-c'   constant differential pressure, `setting` = head in m (capped by the max curve)
      'dp-v'   variable differential pressure: head rises linearly from setting/2 at zero flow to
               `setting` where that line meets the max curve (Wilo/Grundfos 'proportional')
    `on` is a bool or callable(ctx) -> bool; when off the pump is a resistance kv_off (a wet-rotor
    pump lets water through)."""

    curve: PumpCurve
    mode: str = "speed"
    setting: float = 1.0
    on: bool | Callable = True
    kv_off: float = 0.5
    name: str = "pump"

    def is_on(self, ctx) -> bool:
        return bool(self.on(ctx) if callable(self.on) else self.on)

    def head(self, q: float) -> float:
        c = self.curve
        if self.mode == "speed":
            return c.head(q, self.setting)
        if self.mode == "dp-c":
            return min(self.setting, c.head(q))
        if self.mode == "dp-v":
            qs = c.q0 * math.sqrt(max(1 - self.setting / c.H0, 0.0))  # where H = setting on the max curve
            line = self.setting * (0.5 + 0.5 * max(q, 0.0) / max(qs, 1e-9))
            return min(line, c.head(q))
        raise ValueError(f"pump {self.name}: unknown mode {self.mode!r}")

    def dp(self, q, ctx):
        if not self.is_on(ctx):
            return kv_dp(q, self.kv_off)
        return -self.head(q) * KPA_PER_M


# Approximate maximum curves (read off manufacturer charts; replace with your datasheet values).
PUMP_CURVES = {
    # 6 m class wet-rotor circulators used on underfloor manifolds (Wilo Para 25/6, Yonos PICO 25/1-6)
    "wilo-para-25-6": PumpCurve(H0=6.2, q0=3.3, name="Wilo Para 25/6 (approx.)"),
    # 4 m class
    "wilo-para-25-4": PumpCurve(H0=4.2, q0=2.7, name="Wilo Para 25/4 (approx.)"),
}


# ------------------------------------------------------------------ heat transfer


@dataclass
class HeatSource(Element):
    """Heats the water to a set outlet temperature (or by a fixed power), like a heat pump or
    boiler. `t_out` / `power` may be callables(ctx). With both, power is the limit."""

    t_out: float | Callable | None = None
    power: float | Callable | None = None
    dp_kv: float | None = None  # internal resistance (plate exchanger), kv
    name: str = "source"

    def dp(self, q, ctx):
        return kv_dp(q, self.dp_kv) if self.dp_kv else 0.0

    def outlet(self, t_in, q, ctx):
        if abs(q) < 1e-9:
            return t_in
        t_set = self.t_out(ctx) if callable(self.t_out) else self.t_out
        p_max = self.power(ctx) if callable(self.power) else self.power
        t = t_in if t_set is None else max(t_set, t_in)
        if p_max is not None:
            t = min(t, t_in + p_max / w_per_k(q)) if t_set is not None else t_in + p_max / w_per_k(q)
        return t


@dataclass
class Emitter(Element):
    """Gives heat to a space at temperature ctx[space] (C): radiator, floor loop, heat exchanger.
    Effectiveness model: outlet = T_space + (T_in - T_space) exp(-UA / C_dot). UA in W/K, either
    fixed or from a radiator rating (q50 W at dT50, exponent n): UA = q50/49.83^n * dT^(n-1).
    The water-side resistance is a kv (with a dead band: none)."""

    space: str
    UA: float | None = None
    q50: float | None = None
    n: float = 1.3
    kv: float | None = None
    name: str = "emitter"

    def dp(self, q, ctx):
        return kv_dp(q, self.kv) if self.kv else 0.0

    def ua(self, t_in, t_space):
        if self.UA is not None:
            return self.UA
        dT = max(t_in - t_space, 0.5)
        return self.q50 / 49.83**self.n * dT ** (self.n - 1)

    def outlet(self, t_in, q, ctx):
        t_space = ctx[self.space]
        cdot = w_per_k(q)
        if cdot < 1e-6:
            return t_space
        return t_space + (t_in - t_space) * math.exp(-self.ua(t_in, t_space) / cdot)


@dataclass
class Buffer(Element):
    """Fully mixed storage tank of `volume_l` litres in series: outlet = tank temperature, which
    `Circuit.step` advances with the inflow. `loss_ua` W/K to ctx[space] if given. With
    `t_column`, a measured tank temperature in ctx (when present) is used instead of the state."""

    volume_l: float
    t: float = 30.0
    t_column: str | None = None
    space: str | None = None
    loss_ua: float = 0.0
    kv: float | None = None
    name: str = "buffer"

    def dp(self, q, ctx):
        return kv_dp(q, self.kv) if self.kv else 0.0

    def outlet(self, t_in, q, ctx):
        if self.t_column:
            m = ctx.get(self.t_column)
            if m is not None and m == m:
                return float(m)
        return self.t

    def advance(self, inflows: list[tuple[float, float]], dt_s: float, ctx: dict) -> None:
        """inflows: (q m3/h, T) of water entering. Implicit Euler on the tank temperature."""
        cap = self.volume_l * RHO / 1000.0 * CP
        g = sum(w_per_k(q) for q, _ in inflows) + self.loss_ua
        src = sum(w_per_k(q) * t for q, t in inflows) + self.loss_ua * (ctx[self.space] if self.space else 0.0)
        self.t = (cap / dt_s * self.t + src) / (cap / dt_s + g) if g > 0 else self.t


# ------------------------------------------------------------------ branch-level components


@dataclass
class OverflowValve:
    """Differential-pressure overflow (bypass) valve: closed below `p_open` (kPa; 0.4 bar = 40;
    a number or callable(ctx), e.g. a setting that changed on a date), opening linearly over `band`
    kPa to kv_max. Its flow is a function of the pressure across it, so it forms a branch of its
    own (`Circuit.add_overflow`)."""

    p_open: float | Callable
    kv_max: float = 2.0
    band: float = 10.0
    kv_min: float = 1e-3
    name: str = "bypass"

    def flow(self, dp: float, ctx: dict | None = None) -> float:
        p_open = self.p_open(ctx or {}) if callable(self.p_open) else self.p_open
        # smooth opening (softplus) keeps Newton happy at the opening point
        x = (dp - p_open) / self.band
        u = min(math.log1p(math.exp(min(x * 6, 50))) / 6, 1.0)
        kv = self.kv_min + (self.kv_max - self.kv_min) * u
        return kv * math.copysign(math.sqrt(abs(dp) / 100.0), dp)


@dataclass
class ThermostaticMixingValve:
    """Three-way thermostatic mixing valve (wax element) holding the mixed outlet near `t_set`:
    hot port opening u, cold (recirculation) port 1 - u, with a proportional band `band` K:
    u = clip(0.5 + (t_set - T_mixed) / band, 0, 1). Use `hot` and `cold` as the two valve
    elements of the branches into the mixing node; `Circuit.solve` updates u."""

    t_set: float
    kv: float
    band: float = 4.0
    mixed_node: str = ""
    u: float = 0.5
    name: str = "tmv"
    hot: Valve = field(init=False)
    cold: Valve = field(init=False)

    def __post_init__(self):
        self.hot = Valve(self.kv, opening=lambda ctx: self.u, name=f"{self.name}.hot")
        self.cold = Valve(self.kv, opening=lambda ctx: 1.0 - self.u, name=f"{self.name}.cold")

    def target(self, t_mixed: float) -> float:
        return min(max(0.5 + (self.t_set - t_mixed) / self.band, 0.0), 1.0)
