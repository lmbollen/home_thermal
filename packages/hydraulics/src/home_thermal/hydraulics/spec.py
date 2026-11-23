"""Declare a heating circuit in YAML and build it.

    reference: return_main               # node where the expansion vessel connects
    parameters:                          # values referenced as "$name" below; calibratable
      hp_H0: {value: 7.0, bounds: [3, 12]}
    branches:
      - name: heat_pump
        from: buffer_out
        to: hp_out
        elements:
          - {type: pump, curve: {H0: $hp_H0, q0: 2.5}, on: {column: HP_elec_W, above: 150}}
          - {type: heat_source, t_out: {column: T_flow}}
      - {name: bypass, from: hp_out, to: buffer_in, overflow: {p_open: 40, kv_max: 1.5}}
      - mixing_group: {name: ufh_a, from: flow_main, to: return_main, t_set: 30, kv: 2.5,
                       pump: {curve: wilo-para-25-6, mode: dp-c, setting: 2.0, on: {column: UFH_A}},
                       loops: {kv: 1.2, UA: 600, space: slab_woonkamer}}

Values can be numbers, "$parameter" references, or for inputs that change per step
{column: NAME, scale: .., offset: .., above: .., dead_band: ..} read from the step's ctx.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .circuit import Circuit
from .components import (
    PUMP_CURVES,
    Buffer,
    Emitter,
    HeatSource,
    Kv,
    OverflowValve,
    Pipe,
    Pump,
    PumpCurve,
    ThermostaticMixingValve,
    Valve,
)

ELEMENT_TYPES = ("pump", "pipe", "kv", "valve", "heat_source", "emitter", "buffer")


class SpecError(ValueError):
    """The circuit specification is invalid."""


def load_spec(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    if not isinstance(spec, dict) or "branches" not in spec or "reference" not in spec:
        raise SpecError(f"{path}: needs 'reference' and 'branches'")
    return spec


def parameter_values(spec: dict) -> dict:
    return {k: float(v["value"]) for k, v in (spec.get("parameters") or {}).items()}


def build_circuit(spec: dict, values: dict | None = None) -> Circuit:
    """A Circuit from a specification; `values` override parameter values by name."""
    vals = {**parameter_values(spec), **(values or {})}
    unknown = set(values or {}) - set(parameter_values(spec))
    if unknown:
        raise SpecError(f"unknown parameters {sorted(unknown)}")

    def num(v, where):
        if isinstance(v, str) and v.startswith("$"):
            if v[1:] not in vals:
                raise SpecError(f"{where}: unknown parameter {v}")
            return vals[v[1:]]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        raise SpecError(f"{where}: expected a number or $parameter, got {v!r}")

    def signal(v, where, kind="number"):
        """A constant, or a callable(ctx) for {column: ...}."""
        if isinstance(v, dict) and "column" in v:
            col = v["column"]
            scale, offset = num(v.get("scale", 1.0), where), num(v.get("offset", 0.0), where)
            if kind == "bool":
                above = num(v.get("above", 0.5), where)
                return lambda ctx: (ctx.get(col) or 0.0) > above
            dead = num(v.get("dead_band", 0.0), where)

            def f(ctx):
                x = ctx.get(col)
                x = 0.0 if x is None or x != x else float(x)
                if dead:
                    x = max(0.0, (x - dead) / (100.0 - dead)) * 100.0
                return x * scale + offset

            return f
        if kind == "bool":
            return bool(v)
        return num(v, where)

    def curve(v, where):
        if isinstance(v, str):
            if v not in PUMP_CURVES:
                raise SpecError(f"{where}: unknown pump curve {v!r} (known: {', '.join(PUMP_CURVES)})")
            return PUMP_CURVES[v]
        return PumpCurve(num(v["H0"], where), num(v["q0"], where))

    def element(e, where):
        t = e.get("type")
        name = e.get("name", t)
        if t == "pump":
            return Pump(
                curve(e["curve"], where),
                mode=e.get("mode", "speed"),
                setting=num(e.get("setting", 1.0), where),
                on=signal(e.get("on", True), where, "bool"),
                kv_off=num(e.get("kv_off", 0.5), where),
                name=name,
            )
        if t == "pipe":
            return Pipe(num(e["length"], where), num(e["diameter"], where), num(e.get("zeta", 0.0), where), name=name)
        if t == "kv":
            return Kv(num(e["kv"], where), name=name)
        if t == "valve":
            return Valve(
                num(e["kv_max"], where),
                opening=signal(e.get("opening", 1.0), where),
                characteristic=e.get("characteristic", "linear"),
                name=name,
            )
        if t == "heat_source":
            return HeatSource(
                t_out=signal(e["t_out"], where) if "t_out" in e else None,
                power=signal(e["power"], where) if "power" in e else None,
                dp_kv=num(e["kv"], where) if "kv" in e else None,
                name=name,
            )
        if t == "emitter":
            return Emitter(
                e["space"],
                UA=num(e["UA"], where) if "UA" in e else None,
                q50=num(e["q50"], where) if "q50" in e else None,
                n=num(e.get("n", 1.3), where),
                kv=num(e["kv"], where) if "kv" in e else None,
                name=name,
            )
        if t == "buffer":
            return Buffer(
                num(e["volume_l"], where),
                t=num(e.get("t0", 30.0), where),
                t_column=e.get("t_column"),
                space=e.get("space"),
                loss_ua=num(e.get("loss_ua", 0.0), where),
                kv=num(e["kv"], where) if "kv" in e else None,
                name=name,
            )
        raise SpecError(f"{where}: element type must be one of {ELEMENT_TYPES}, got {t!r}")

    c = Circuit(spec["reference"])
    for i, b in enumerate(spec["branches"]):
        where = f"branches[{i}]"
        if "mixing_group" in b:
            g = b["mixing_group"]
            n, w = g["name"], f"{where}.mixing_group"
            tmv = ThermostaticMixingValve(
                num(g["t_set"], w), num(g["kv"], w), num(g.get("band", 4.0), w), name=f"{n}.tmv"
            )
            c.add_mixing_valve(tmv, hot_from=g["from"], cold_from=f"{n}.loop_ret", mixed=f"{n}.mix")
            c.add(
                f"{n}.pump", f"{n}.mix", f"{n}.loop", [element({"type": "pump", **g["pump"], "name": f"{n}.pump"}, w)]
            )
            loops = g["loops"] if isinstance(g["loops"], list) else [g["loops"]]
            for k, lp in enumerate(loops):  # parallel loop groups, e.g. one per room
                tag = lp.get("name", f"loops{k}" if len(loops) > 1 else "loops")
                c.add(
                    f"{n}.{tag}",
                    f"{n}.loop",
                    f"{n}.loop_ret",
                    [
                        Kv(num(lp["kv"], w), name=f"{n}.{tag}.kv"),
                        Emitter(lp["space"], UA=num(lp["UA"], w), name=f"{n}.{tag}"),
                    ],
                )
            c.add(
                f"{n}.primary_return", f"{n}.loop_ret", g["to"], [Kv(num(g.get("return_kv", 5.0), w), name=f"{n}.ret")]
            )
            continue
        if "overflow" in b:
            o = b["overflow"]
            c.add_overflow(
                b["name"],
                b["from"],
                b["to"],
                OverflowValve(
                    signal(o["p_open"], where),
                    num(o.get("kv_max", 2.0), where),
                    num(o.get("band", 10.0), where),
                    name=b["name"],
                ),
            )
            continue
        c.add(
            b["name"], b["from"], b["to"], [element(e, f"{where}.elements[{k}]") for k, e in enumerate(b["elements"])]
        )
    return c
