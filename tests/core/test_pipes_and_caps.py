"""Heating pipes through rooms (emitters.pipes) and the mixing-valve cap of a floor circuit
(floor_heating.<z>.t_max)."""
import numpy as np
import pandas as pd
import pytest
import yaml

from home_thermal.core import (
    ConfigError,
    Network,
    emitter_heat,
    energy_balance,
    floor_caps,
    normalise,
    parameter_set,
    simulate,
    simulate_detailed,
)
from home_thermal.core.synthetic import demo_house_yaml


def house(pipes=None, t_max=None):
    raw = yaml.safe_load(demo_house_yaml())
    if pipes is not None:
        raw["emitters"]["pipes"] = pipes
    if t_max is not None:
        raw["emitters"]["floor_heating"]["living"]["t_max"] = t_max
    return normalise(raw)


# ---------------------------------------------------------------- pipes

def test_pipe_parameter_from_length_and_insulation():
    p = parameter_set(house({"bedroom": {"length_m": 8, "W_per_mK": 0.5}}))["UA_pipe_bedroom"]
    assert p.prior == pytest.approx(4.0) and p.bounds == (0.0, 20.0)
    assert not p.fixed                                  # identifiable: the room warms with its valve shut
    assert not any(n.startswith("UA_pipe_") for n in parameter_set(house()))


@pytest.mark.parametrize("pipes, message", [
    ({"attic": {"length_m": 3, "W_per_mK": 0.5}}, "attic"),
    ({"bedroom": {"length_m": 3}}, "W_per_mK: missing"),
    ({"bedroom": {"length_m": 3, "W_per_mk": 0.5}}, "did you mean 'W_per_mK'"),
])
def test_bad_pipes_are_named(pipes, message):
    with pytest.raises(ConfigError, match=message):
        house(pipes)


def test_pipe_heat_goes_to_the_room_and_counts_as_source_heat():
    net = Network(house({"bedroom": {"length_m": 4, "W_per_mK": 1.0}}))
    plain = Network(house())
    x = np.full(net.n, 18.0)
    row = {"T_water": 38.0, "T_flow": 40.0, "circ": 1.0, "T_out": 0.0, "P_living": 0.0,
           "V_bedroom": 0.0, "V_living": 0.0}
    q, src = emitter_heat(net, x, row)
    q0, src0 = emitter_heat(plain, x, row)
    i = net.iA["bedroom"]
    assert q[i] - q0[i] == pytest.approx(4.0 * 20.0)
    assert src - src0 == pytest.approx(80.0)
    assert emitter_heat(net, x, {**row, "circ": 0.0})[0][i] == 0.0          # no circulation, no heat


def test_energy_balance_still_closes_with_pipes(data):
    net = Network(house({"bedroom": {"length_m": 4, "W_per_mK": 1.0}}))
    eb = energy_balance(net, simulate_detailed(net, data))
    assert any("pipes bedroom" in s for s in eb["sources"].source)
    supplied = eb["sources"].query("group not in ('hot water', 'info')").kWh.sum()
    assert supplied - eb["losses"].kWh.sum() == pytest.approx(eb["stored_kWh"], abs=0.02 * supplied)


# ---------------------------------------------------------------- floor caps

def test_t_max_number_or_schedule():
    assert house(t_max=30)["emitters"]["floor_heating"]["living"]["t_max"] == 30.0
    sched = house(t_max={"2026-11-01": 30, "2026-11-25T18:15:00Z": 40})["emitters"]["floor_heating"]["living"]["t_max"]
    assert [c for _, c in sched] == [30.0, 40.0] and str(sched[0][0].tz) == "UTC"
    with pytest.raises(ConfigError, match="not a time"):
        house(t_max={"yesterday-ish": 30})
    with pytest.raises(ConfigError, match="t_max"):
        house(t_max="hot")


def test_floor_caps_column():
    idx = pd.date_range("2026-11-25 17:00", periods=8, freq="30min", tz="UTC")
    cfg = house(t_max={"2026-11-25T18:15:00Z": 40, "2026-11-25T17:30:00Z": 30})
    col = floor_caps(cfg, pd.DataFrame(index=idx))["t_max:living"]
    assert np.isnan(col.iloc[0])                                    # before the first setting: no cap
    assert col.iloc[1:3].tolist() == [30.0, 30.0] and col.iloc[3:].eq(40.0).all()
    assert floor_caps(house(), pd.DataFrame(index=idx)).columns.tolist() == []


def test_cap_limits_floor_heat_only_above_the_setting():
    net = Network(house(t_max=30))
    free = Network(house())
    x = np.full(net.n, 20.0)
    s = net.iS["living"]
    for tw, same in ((28.0, True), (40.0, False)):
        row = {"T_water": tw, "T_flow": tw + 2, "circ": 1.0, "T_out": 0.0, "P_living": 1.0, "t_max:living": 30.0,
               "V_bedroom": 0.0, "V_living": 0.0}
        no_cap = {k: v for k, v in row.items() if k != "t_max:living"}     # floor_caps adds it only with t_max
        capped, open_ = emitter_heat(net, x, row)[0][s], emitter_heat(free, x, no_cap)[0][s]
        assert (capped == pytest.approx(open_)) if same else (capped == pytest.approx(open_ * 10 / 20))


def test_simulate_applies_the_cap(data):
    hot = data.copy()
    hot["T_water"] = hot["T_water"] + 15.0
    _, q_cap = simulate(Network(house(t_max=25)), hot)
    _, q_free = simulate(Network(house()), hot)
    assert q_cap.sum() < q_free.sum()
