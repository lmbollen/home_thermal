import numpy as np
import pandas as pd
import pytest

from home_thermal.core import Network, check_inputs, energy_balance, normalise, simulate, simulate_detailed


def one_zone(**zone):
    return normalise({
        "zones": {"room": {"area": 20, "volume": 50, "ach": 0.5, "sensor": "T_room", **zone}},
        "envelope": {"room": [{"kind": "wall", "area": 30, "U": 0.5}, {"kind": "window", "area": 4, "U": 2.0}]},
    })


def test_conductance_matrix_is_consistent(cfg):
    net = Network(cfg)
    assert np.allclose(net.G, net.G.T)
    # each row of G sums to that node's conductance to the boundaries
    assert np.allclose(net.G.sum(axis=1), net.Gb.sum(axis=1))
    assert (net.C > 0).all()


def test_house_loss_coefficient_by_hand():
    net = Network(one_zone())
    vent = 0.5 * 50 * 1200 / 3600
    assert net.H_house() == pytest.approx(30 * 0.5 + 4 * 2.0 + vent)
    assert Network(one_zone(in_house_loss=False)).H_house() == 0


def test_multipliers_scale_the_right_paths():
    cfg = one_zone()
    base, scaled = Network(cfg), Network(cfg, {"m_env": 2.0})
    assert scaled.H_parts["opaque"] == pytest.approx(2 * base.H_parts["opaque"])
    assert scaled.H_parts["windows"] == pytest.approx(base.H_parts["windows"])


def test_steady_state_matches_heat_loss():
    """Constant 500 W into one zone settles at T_out + 500 / H."""
    net = Network(one_zone(), {"Q_int": 500.0})
    idx = pd.date_range("2026-01-01", periods=24 * 4 * 30, freq="15min", tz="UTC")
    df = pd.DataFrame({"T_out": 0.0, "T_room": 15.0}, index=idx)
    X, Q = simulate(net, df)
    assert X["air:room"].iloc[-1] == pytest.approx(500 / net.H_house(), rel=1e-3)
    assert (Q == 0).all()


def test_energy_is_conserved(cfg, data):
    net = Network(cfg)
    eb = energy_balance(net, simulate_detailed(net, data))
    supplied = eb["sources"].query("group != 'hot water'").kWh.sum()
    lost = eb["losses"].kWh.sum()
    assert supplied - lost == pytest.approx(eb["stored_kWh"], abs=0.02 * supplied)


def test_bad_data_is_rejected(cfg, data):
    with pytest.raises(ValueError, match="missing columns"):
        check_inputs(cfg, data.drop(columns=["V_bedroom"]))
    with pytest.raises(ValueError, match="regular"):
        check_inputs(cfg, data.drop(index=data.index[5]))
