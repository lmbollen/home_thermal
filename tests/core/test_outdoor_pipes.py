"""Outdoor pipes of a monoblock heat pump: heat the unit produces (and meters) that never reaches
the house."""
import pytest
import yaml

from home_thermal.core import (
    ConfigError,
    Network,
    energy_balance,
    normalise,
    outdoor_pipe_loss,
    parameter_set,
    simulate,
    simulate_detailed,
)
from home_thermal.core.synthetic import demo_house_yaml


def house(pipes=None, **extra):
    raw = yaml.safe_load(demo_house_yaml())
    if pipes is not None:
        raw["heat_pump"] = {"outdoor_pipes": pipes}
    raw.update(extra)
    return normalise(raw)


def test_prior_comes_from_length_and_insulation():
    p = parameter_set(house({"length_m": 6, "W_per_mK": 0.5}))["UA_pipe_out"]
    assert p.prior == pytest.approx(3.0)
    assert p.fixed                                   # not identifiable: held unless freed
    assert p.bounds == (0.0, 15.0)
    assert "UA_pipe_out" not in parameter_set(house())


@pytest.mark.parametrize("pipes, message", [
    ({"length_m": 6}, "W_per_mK: missing"),
    ({"length_m": -1, "W_per_mK": 0.5}, "length_m: must be > 0"),
    ({"lengt_m": 6, "W_per_mK": 0.5}, "did you mean 'length_m'"),
    (5, "expected a mapping"),
])
def test_bad_pipe_config_is_named(pipes, message):
    with pytest.raises(ConfigError, match=message):
        house(pipes)


def test_not_with_a_boiler():
    with pytest.raises(ConfigError, match="one heat source"):
        house({"length_m": 6, "W_per_mK": 0.5}, boiler={})


def test_loss_only_while_circulating():
    net = Network(house({"length_m": 4, "W_per_mK": 1.0}))
    assert outdoor_pipe_loss(net, {"T_water": 30.0, "T_out": 0.0, "circ": 1.0}) == pytest.approx(120.0)
    assert outdoor_pipe_loss(net, {"T_water": 30.0, "T_out": 0.0, "circ": 0.5}) == pytest.approx(60.0)
    assert outdoor_pipe_loss(net, {"T_water": float("nan"), "T_out": 0.0, "circ": 0.0}) == 0.0
    assert outdoor_pipe_loss(Network(house()), {"T_water": 30.0, "T_out": 0.0, "circ": 1.0}) == 0.0


def test_source_heat_includes_pipes_rooms_do_not(data):
    with_pipes, without = Network(house({"length_m": 4, "W_per_mK": 1.0})), Network(house())
    Xa, Qa = simulate(with_pipes, data)
    Xb, Qb = simulate(without, data)
    assert (Xa - Xb).abs().to_numpy().max() < 1e-9  # the house sees no difference
    tw = data["T_water"].where(data["circ"] > 0)
    expect = (4.0 * (tw - data["T_out"]).clip(lower=0) * data["circ"]).fillna(0)
    assert (Qa - Qb).to_numpy() == pytest.approx(expect.to_numpy(), abs=1e-6)

    eb = energy_balance(with_pipes, simulate_detailed(with_pipes, data))
    row = eb["sources"].query("source == 'heat source: lost by the outdoor pipes'")
    assert row.group.tolist() == ["info"] and row.kWh.iloc[0] == pytest.approx(expect.sum() * 0.25 / 1000)
    supplied = eb["sources"].query("group not in ('hot water', 'info')").kWh.sum()
    assert supplied - eb["losses"].kWh.sum() == pytest.approx(eb["stored_kWh"], abs=0.02 * supplied)
