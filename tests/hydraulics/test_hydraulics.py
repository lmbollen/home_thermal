import math

import pytest

from home_thermal.hydraulics import (
    KPA_PER_M,
    Buffer,
    Circuit,
    Emitter,
    HeatSource,
    Kv,
    OverflowValve,
    Pipe,
    Pump,
    PumpCurve,
    ThermostaticMixingValve,
    Valve,
    kv_dp,
)

CURVE = PumpCurve(H0=6.0, q0=3.0)


def loop(load_kv=1.0, mode="speed", setting=1.0):
    c = Circuit("ret")
    c.add("hp", "ret", "flow", [Pump(CURVE, mode=mode, setting=setting), HeatSource(t_out=40)])
    c.add("load", "flow", "ret", [Kv(load_kv), Emitter("room", UA=100)])
    return c


def test_operating_point_matches_hand_calculation():
    q = loop().solve({"room": 20}).flow["load"]
    # 6 (1 - q^2/9) m x 9.74 kPa/m = 100 q^2 kPa
    expect = math.sqrt(6 * KPA_PER_M / (100 + 6 * KPA_PER_M / 9))
    assert q == pytest.approx(expect, rel=1e-6)


def test_heat_balances():
    r = loop().solve({"room": 20})
    assert r.heat["source"] == pytest.approx(-r.heat["emitter"], rel=1e-6)
    assert 40 > r.temperature["ret"] > 20


def test_parallel_branches_split_by_resistance():
    c = Circuit("ret")
    c.add("hp", "ret", "flow", [Pump(CURVE)])
    c.add("a", "flow", "ret", [Kv(1.0)])
    c.add("b", "flow", "ret", [Kv(2.0)])
    r = c.solve({})
    assert r.flow["b"] == pytest.approx(2 * r.flow["a"], rel=1e-6)  # same dp, q ~ kv
    assert r.flow["hp"] == pytest.approx(r.flow["a"] + r.flow["b"], rel=1e-9)


def test_pump_modes():
    q_c = [loop(kv, "dp-c", 3.0).solve({"room": 20}) for kv in (0.5, 2.0)]
    heads = [r.dp("flow", "ret") / KPA_PER_M for r in q_c]
    assert heads[0] == pytest.approx(3.0, rel=1e-3) and heads[1] == pytest.approx(3.0, rel=1e-3)  # constant head
    q_v = [loop(kv, "dp-v", 4.0).solve({"room": 20}).dp("flow", "ret") / KPA_PER_M for kv in (0.3, 1.2)]
    assert q_v[0] < q_v[1] < 4.0  # head falls at low flow
    slow = loop(1.0, "speed", 0.6).solve({"room": 20}).flow["load"]
    assert slow < loop().solve({"room": 20}).flow["load"]


def test_pump_off_is_a_resistance():
    c = Circuit("ret")
    c.add("main", "ret", "flow", [Pump(CURVE)])
    c.add("other", "ret", "flow", [Pump(CURVE, on=lambda ctx: ctx["on"], kv_off=0.5)])
    c.add("load", "flow", "ret", [Kv(1.0)])
    r = c.solve({"on": False})
    assert r.flow["other"] < 0  # water flows back through the stopped pump


def test_overflow_valve_opens_above_setting():
    def circuit(valve_open):
        c = Circuit("ret")
        c.add("hp", "ret", "flow", [Pump(CURVE)])
        c.add("load", "flow", "ret", [Valve(3.0, opening=valve_open)])
        c.add_overflow("bypass", "flow", "ret", OverflowValve(p_open=40, kv_max=2.0, band=10))
        return c.solve({})

    open_, closed = circuit(1.0), circuit(0.02)
    assert open_.dp("flow", "ret") < 40 and open_.flow["bypass"] < 0.02 * open_.flow["load"]
    assert closed.dp("flow", "ret") > 40 and closed.flow["bypass"] > 0.5  # takes over the flow


def test_mixing_valve_holds_its_setpoint():
    """Hot supply at 45 C, floor loop to a 22 C slab: the mixed supply settles near 30 C, within
    the valve's proportional band."""
    tmv = ThermostaticMixingValve(t_set=30.0, kv=2.0, band=4.0)
    c = Circuit("ret")
    c.add("hp", "ret", "sup", [Pump(CURVE), HeatSource(t_out=45)])
    c.add_mixing_valve(tmv, hot_from="sup", cold_from="loop_ret", mixed="mix")
    c.add("ufh_pump", "mix", "loop", [Pump(PumpCurve(6.2, 3.3), mode="dp-c", setting=2.0)])
    c.add("loops", "loop", "loop_ret", [Kv(1.5), Emitter("slab", UA=600)])
    c.add("prim_ret", "loop_ret", "ret", [Kv(3.0)])
    r = c.solve({"slab": 22.0})
    u = tmv.u
    assert 0 < u < 1  # regulating, not saturated
    assert r.temperature["mix"] == pytest.approx(30.0 + 4.0 * (0.5 - u), abs=0.01)  # wax valve: P-band
    assert 26.0 < r.temperature["mix"] < 32.0
    assert r.flow["tmv.hot"] < r.flow["ufh_pump"]  # only part of the loop flow is fresh supply
    assert r.heat["source"] == pytest.approx(-r.heat["emitter"], rel=1e-4)


def test_buffer_warms_toward_inflow():
    buf = Buffer(volume_l=50, t=20.0)
    c = Circuit("ret")
    c.add("hp", "ret", "flow", [Pump(CURVE), HeatSource(t_out=40)])
    c.add("load", "flow", "ret", [Kv(1.0), buf])
    temps = [buf.t]
    for _ in range(20):
        c.step({}, 60.0)
        temps.append(buf.t)
    assert all(b > a for a, b in zip(temps, temps[1:])) and temps[-1] < 40.0 + 1e-9
    q = c.solve({}).flow["load"]
    tau = 50 * 0.993 / (q * 993 / 3600)  # volume / volumetric flow, s
    assert temps[1] == pytest.approx(40 - 20 * (1 / (1 + 60 / tau)), rel=1e-3)  # implicit Euler step


def test_pipe_friction_reasonable():
    # 10 m of 22 mm copper (20 mm inside) at 0.8 m3/h (0.7 m/s): about 330 Pa/m
    assert 2.5 < Pipe(10, 0.020).dp(0.8, {}) < 4.5
    assert Pipe(10, 0.020).dp(-0.8, {}) == pytest.approx(-Pipe(10, 0.020).dp(0.8, {}))
    assert kv_dp(1.0, 1.0) == 100.0
