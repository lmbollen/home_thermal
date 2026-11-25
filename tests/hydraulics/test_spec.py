from pathlib import Path

import numpy as np
import pytest
import yaml

from home_thermal.hydraulics import SpecError, build_circuit, load_spec
from home_thermal.hydraulics.calibrate import fit_circuit, predict_rows

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "semi_detached_1935" / "hydraulics.yaml"

SPEC = yaml.safe_load("""
reference: ret
parameters:
  H0:  {value: 6.0, bounds: [2, 12]}
  kv1: {value: 1.0, bounds: [0.2, 5]}
branches:
  - name: hp
    from: ret
    to: flow
    elements:
      - {type: pump, curve: {H0: $H0, q0: 3.0}, on: {column: run}}
      - {type: heat_source, t_out: {column: T_flow}}
  - name: rad
    from: flow
    to: ret
    elements:
      - {type: valve, kv_max: $kv1, opening: {column: V, dead_band: 20, scale: 0.01}}
      - {type: emitter, space: T_room, q50: 1500}
  - {name: bypass, from: flow, to: ret, overflow: {p_open: 40, kv_max: 1.0}}
  - mixing_group:
      name: ufh
      from: flow
      to: ret
      t_set: 30
      kv: 1.5
      pump: {curve: wilo-para-25-6, mode: dp-c, setting: 2.0, on: true}
      loops: [{name: a, kv: 1.0, UA: 300, space: slab}, {name: b, kv: 0.8, UA: 200, space: slab}]
""")


def test_spec_builds_and_solves():
    c = build_circuit(SPEC)
    res = c.solve({"run": 1, "T_flow": 40.0, "V": 60, "T_room": 20.0, "slab": 24.0})
    assert res.flow["hp"] > 0
    assert set(res.flow) >= {"ufh.a", "ufh.b", "ufh.tmv.hot", "bypass"}
    assert res.temperature["flow"] == pytest.approx(40.0)
    assert res.temperature["ufh.mix"] < 34


def test_dead_band_closes_the_valve():
    c = build_circuit(SPEC)
    shut = c.solve({"run": 1, "T_flow": 40.0, "V": 15, "T_room": 20.0, "slab": 24.0})
    assert abs(shut.flow["rad"]) < 0.01


def test_spec_errors_are_clear():
    bad = {**SPEC, "branches": [{"name": "x", "from": "a", "to": "ret", "elements": [{"type": "pumpp"}]}]}
    with pytest.raises(SpecError, match="element type"):
        build_circuit(bad)
    with pytest.raises(SpecError, match="unknown parameters"):
        build_circuit(SPEC, {"nope": 1.0})
    bad = {**SPEC, "branches": [{"name": "x", "from": "a", "to": "ret", "elements": [{"type": "kv", "kv": "$zz"}]}]}
    with pytest.raises(SpecError, match=r"\$zz"):
        build_circuit(bad)


def test_fit_recovers_known_parameters():
    truth = {"H0": 8.0, "kv1": 1.8}
    rng = np.random.default_rng(0)
    ctxs = [
        {"run": 1, "T_flow": float(t), "V": float(v), "T_room": 20.0, "slab": 24.0}
        for t, v in zip(rng.uniform(30, 45, 25), rng.uniform(25, 100, 25))
    ]
    predict = lambda res: res.flow["hp"]  # noqa: E731
    obs = predict_rows(SPEC, truth, ctxs, predict)
    fit = fit_circuit(SPEC, ctxs, obs, predict, sigma_obs=0.01)
    assert fit.values["H0"] == pytest.approx(8.0, rel=0.05)
    assert fit.values["kv1"] == pytest.approx(1.8, rel=0.05)


def test_example_installation_builds():
    spec = load_spec(EXAMPLE)
    c = build_circuit(spec)
    ctx = {
        "HP_elec_W": 900,
        "T_flow": 34.0,
        "T_return": 30.0,
        "bypass_kpa": 40.0,
        "UFH_A": 1,
        "UFH_B": 0,
        "DHW_W": 0,
        **{f"V_{z}": 40.0 for z in ("slaapkamer", "hobbykamer", "inloopkast", "zolder", "badkamer")},
        **{f"T_{z}": 19.0 for z in ("slaapkamer", "hobbykamer", "inloopkast", "zolder", "badkamer", "berging")},
        **{f"slab_{z}": 23.0 for z in ("woonkamer", "bijkeuken", "berging", "badkamer")},
    }
    res = c.solve(ctx)
    assert res.flow["heat_pump"] > 0.1
    assert sum(res.flow[b] for b in ("main", "bypass")) == pytest.approx(res.flow["heat_pump"], rel=1e-6)


def test_pump_on_condition_survives_yaml():
    # PyYAML reads the bare key `on:` as True; the pump must still follow its column
    c = build_circuit(SPEC)
    hp = c.branches["hp"].elements[0]
    assert callable(hp.on)
    assert hp.is_on({"run": 1}) and not hp.is_on({"run": 0})
    stopped = c.solve({"run": 0, "T_flow": 40.0, "V": 60, "T_room": 20.0, "slab": 24.0})
    assert abs(stopped.flow["hp"]) < 0.3        # only what the ufh pump pulls through the stopped one


def test_example_pumps_follow_their_columns():
    c = build_circuit(load_spec(EXAMPLE))
    for br, col in (("heat_pump", "HP_elec_W"), ("edel", "DHW_W")):
        pump = c.branches[br].elements[0]
        assert callable(pump.on), br
        assert not pump.is_on({col: 0})
