from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from home_thermal.calibrate import data_terms, fit, score
from home_thermal.core import ConfigError, Network, default_free, dhw_steps, gas_power, load_house, normalise
from home_thermal.core.synthetic import synthetic_data

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "apartment_gas_boiler" / "house.yaml"
TRUTH = {"eta_boiler": 0.9, "m_env": 1.3, "m_vent": 1.3}


@pytest.fixture(scope="module")
def cfg():
    return load_house(EXAMPLE)


@pytest.fixture(scope="module")
def data(cfg):
    return synthetic_data(cfg, TRUTH, days=8, solar=False, seed=3)


def test_boiler_parameters(cfg):
    free = default_free(cfg)
    assert "eta_boiler" not in free                             # fixed: set from the boiler type
    assert "eta_hp" not in free and "P_aux_hp" not in free      # no heat pump here
    assert "UA_boiler" not in free                              # boiler has no zone


def test_boiler_config_errors(cfg):
    raw = {"zones": {"a": {"area": 10, "volume": 25, "sensor": "T"}}}
    with pytest.raises(ConfigError, match="one heat source"):
        normalise({**raw, "heat_pump": {"standby_W": 5}, "boiler": {}})
    with pytest.raises(ConfigError, match="gas_unit"):
        normalise({**raw, "boiler": {"gas_unit": "litre"}})
    with pytest.raises(ConfigError, match="did you mean 'gas_column'"):
        normalise({**raw, "boiler": {"gas_colum": "g"}})
    assert normalise({**raw, "boiler": None})["boiler"] is None          # 'boiler:' left empty = none


def test_gas_units_agree(cfg, data):
    w = gas_power(cfg, data)
    as_kwh = normalise({**cfg, "boiler": {**cfg["boiler"], "gas_unit": "kWh", "gas_column": "g"}})
    kwh = data.assign(g=data["gas_m3"] * cfg["boiler"]["kwh_per_m3"])
    assert np.allclose(gas_power(as_kwh, kwh), w)


def test_tap_water_steps_detected(cfg, data):
    flagged = dhw_steps(cfg, data)
    assert flagged.sum() == (data["dhw_active"] > 0).sum() > 0
    # without the flag column, gas while the heating does not circulate is still recognised
    no_flag = normalise({**cfg, "boiler": {**cfg["boiler"], "dhw_flag": None}})
    assert (dhw_steps(no_flag, data) >= flagged).all()


def test_gas_meter_pins_heat_loss(cfg, data):
    """With the efficiency known, the gas meter pins the heat demand: H_house is recovered even
    though the truth (envelope and ventilation 30 % worse) is far from the priors."""
    assert "gas" in data_terms(cfg, data)
    h_true, h_prior = Network(cfg, TRUTH).H_house(), Network(cfg).H_house()
    res = fit(cfg, data)
    h_fit = Network(cfg, res.params).H_house()
    assert abs(h_fit - h_true) < 0.25 * abs(h_prior - h_true)
    no_gas = fit(cfg, data.drop(columns=["gas_m3"]))
    assert abs(h_fit - h_true) < abs(Network(cfg, no_gas.params).H_house() - h_true)
    sc = score(cfg, res.params, data)
    assert sc.loc["gas (W, hourly)", "kind"] == "heat source"
    assert sc.loc["gas (W, hourly)", "rmse"] < 100


def test_without_gas_data_efficiency_is_not_fitted(cfg, data):
    res = fit(cfg, data.drop(columns=["gas_m3"]), free=["m_env", "eta_boiler"])   # explicitly freed
    assert res.free == ["m_env"]


def test_hass_gas_counter(tmp_path):
    import sqlite3

    from home_thermal.hass import HassConfig, load_states
    db = tmp_path / "ha.db"
    con = sqlite3.connect(db)
    con.executescript("""
        create table states_meta (metadata_id integer primary key, entity_id text);
        create table states (state_id integer primary key, metadata_id int, state text,
                             last_updated_ts real, attributes_id int);""")
    con.execute("insert into states_meta values (1, 'sensor.gas_total')")
    t0 = pd.Timestamp("2026-11-10", tz="UTC").timestamp()
    for k in range(9):                                     # +0.05 m3 every 15 minutes
        con.execute("insert into states (metadata_id, state, last_updated_ts) values (1, ?, ?)",
                    (str(100 + 0.05 * k), t0 + k * 900))
    con.commit()
    con.close()
    df = load_states(db, HassConfig(entities={"gas_total": "sensor.gas_total"}, gas_counter="gas_total"))
    assert df["gas_m3"].iloc[1:].round(6).eq(0.05).all()
