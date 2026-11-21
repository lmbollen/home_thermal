"""Regression tests for bugs found by the independent adversarial review (Nov 2026)."""
import copy
import sqlite3

import numpy as np
import pandas as pd
import pytest
import yaml

from home_thermal.core import (
    ConfigError,
    Network,
    energy_balance,
    normalise,
    parameter_set,
    simulate,
    simulate_detailed,
)
from home_thermal.core.physics import sensor_reading, valve_sensor_warming
from home_thermal.core.synthetic import demo_house, demo_house_yaml, synthetic_data


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load(demo_house_yaml())


def cfg_with(raw, f):
    c = copy.deepcopy(raw)
    f(c)
    return normalise(c)


# ---------------- 1. boundaries section is silently ignored ----------------
def test_boundaries_T_ground_and_T_neighbour_are_used(raw):
    def f(c):
        c["boundaries"] = {"T_ground": 2.0, "T_neighbour": 5.0}
        c["parameters"] = {"T_ground": {"fixed": True}, "T_neighbour": {"fixed": True}}
    net = Network(cfg_with(raw, f))
    # docs/configuration.md: boundaries 'used only when not fitted as parameters'
    assert net.boundary_temps(0.0).tolist() == [0.0, 2.0, 5.0]   # observed [0, 11, 16]


# ---------------- 2. energy balance with a hot-water tank ----------------
def _dhw_cfg(raw, model="ntu"):
    def f(c):
        c["emitters"]["radiator_model"] = model
        c["dhw"] = {"zone": "hall", "volume_l": 200, "power": "DHW_W"}
    return cfg_with(raw, f)


@pytest.mark.parametrize("model", ["ntu", "catalogue"])
def test_dhw_heat_into_tank_respects_T_max(raw, model):
    cfg = _dhw_cfg(raw, model)
    df = synthetic_data(cfg, days=3)
    df["DHW_W"] = 500.0                     # runs all the time -> tank reaches T_max quickly
    net = Network(cfg)
    sim = simulate_detailed(net, df)
    eb = energy_balance(net, sim)
    reported = eb["sources"].set_index("source").loc["heat into tank", "kWh"]
    h = sim["dt"] / 3.6e6
    tap = np.where(sim["X"][:, net.iT] > cfg["dhw"]["T_cold"] + 5, net.p["tap_W"], 0.0)
    actual = (sim["Qe"][:, net.iT] + tap).sum() * h       # what dhw_heat really put in the tank
    assert reported == pytest.approx(actual, rel=0.05)


def test_dhw_tap_draw_appears_in_balance(raw):
    cfg = _dhw_cfg(raw)
    df = synthetic_data(cfg, days=3)          # normal 1 h/day of DHW
    net = Network(cfg)
    sim = simulate_detailed(net, df)
    eb = energy_balance(net, sim)
    into_nodes = eb["sources"][eb["sources"].group != "info"].kWh.sum()
    out = eb["losses"].kWh.sum() + eb["stored_kWh"]
    assert "taps" in set(eb["losses"]["to"])           # tap draw is a loss path
    assert into_nodes == pytest.approx(out, abs=1.0)


# ---------------- 3. catalogue sensor_on_valve ignored by filter / controller ----------------
def _on_valve(raw, model):
    def f(c):
        c["emitters"]["radiator_model"] = model
        c["emitters"]["radiators"]["bedroom"]["sensor_on_valve"] = True
    return cfg_with(raw, f)


def test_valve_sensor_offset_catalogue_consistent_with_sensor_reading(raw):
    cfg = _on_valve(raw, "catalogue")
    p = {"kappa_s": 0.3}
    df = synthetic_data(cfg, p, days=1, noise_K=0.0)
    net = Network(cfg, p)
    k = int(np.argmax(df["V_bedroom"].to_numpy()))
    t_air = 19.0
    expect = sensor_reading(net, "bedroom", np.array([t_air]), df.iloc[[k]])[0] - t_air
    assert expect > 0.5
    assert valve_sensor_warming(net, "bedroom", df.iloc[k].to_dict(), t_air) == pytest.approx(expect)


def test_kalman_unbiased_with_catalogue_valve_sensor(raw):
    from home_thermal.calibrate import kalman
    cfg = _on_valve(raw, "catalogue")
    p = {"kappa_s": 0.3}
    df = synthetic_data(cfg, p, days=3, noise_K=0.0)   # noise-free, true parameters
    net = Network(cfg, p)
    X, _ = simulate(net, df)
    XS, DS = kalman(net, df)
    err = (XS["air:bedroom"] - X["air:bedroom"]).iloc[50:].abs().mean()
    assert err < 0.3          # observed ~1.6 K, plus a fictitious ~+126 W unexplained heat in bedroom


# ---------------- 4. fit: log-scale parameter with bounds None ----------------
def test_fit_log_param_unbounded(raw):
    from home_thermal.calibrate import fit
    cfg = cfg_with(raw, lambda c: c.update(parameters={"m_env": {"bounds": None}}))
    df = synthetic_data(cfg, days=3)
    res = fit(cfg, df, free=["m_env"])   # observed: ValueError 'Initial guess is outside of provided bounds'
    assert 0.5 < res.params["m_env"] < 2


# ---------------- 5/6. hass statistics ----------------
T0 = pd.Timestamp("2026-11-10 00:00", tz="UTC").timestamp()


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "ha.db"
    con = sqlite3.connect(path)
    con.executescript("""
        create table states_meta (metadata_id integer primary key, entity_id text);
        create table state_attributes (attributes_id integer primary key, shared_attrs text);
        create table states (state_id integer primary key, metadata_id int, state text,
                             last_updated_ts real, attributes_id int);
        create table statistics_meta (id integer primary key, statistic_id text, unit_of_measurement text);
        create table statistics (id integer primary key, metadata_id int, start_ts real,
                                 mean real, min real, sum real);""")
    con.executemany("insert into states_meta values (?,?)", [(1, "sensor.t_room"), (2, "sensor.heat_kw")])
    for k in range(24 * 6):                                     # states: day 3, every 10 min
        t = T0 + 2 * 86400 + k * 600
        con.execute("insert into states (metadata_id,state,last_updated_ts) values (1,'20.0',?)", (t,))
        con.execute("insert into states (metadata_id,state,last_updated_ts) values (2,'2.0',?)", (t,))
    con.executemany("insert into statistics_meta values (?,?,?)",
                    [(1, "sensor.t_room", "°C"), (2, "sensor.heat_kw", "kW")])
    for h in range(48):
        if 20 <= h < 30:                                        # 10-hour recorder outage
            continue
        t = T0 + h * 3600
        con.execute("insert into statistics (metadata_id,start_ts,mean,min) values (1,?,?,?)", (t, 19 + 0.01 * h, 18))
        con.execute("insert into statistics (metadata_id,start_ts,mean,min) values (2,?,1.5,1.0)", (t,))
    con.commit()
    con.close()
    return path


@pytest.fixture
def hc():
    from home_thermal.hass import HassConfig
    return HassConfig(entities={"T_room": "sensor.t_room", "HP_heat_W": "sensor.heat_kw"},
                      statistics_start="2026-11-10", kw_columns=["HP_heat_W"])


def test_statistics_gap_is_not_forward_filled(db, hc):
    from home_thermal.hass import load_statistics
    st = load_statistics(db, hc)
    # observed: the 10 h outage (20:00-06:00) is filled with the 19:00 value
    assert np.isnan(st.loc["2026-11-10 22:00", "T_room"])


def test_kw_units_consistent_between_states_and_statistics(db, hc):
    from home_thermal.hass import load_full
    df = load_full(db, hc)
    stats_part = df.loc[df.from_stats == 1, "HP_heat_W"].dropna()
    states_part = df.loc[df.from_stats == 0, "HP_heat_W"].dropna()
    # statistics are converted by their unit, states by kw_columns
    assert stats_part.median() > 100
    assert states_part.median() > 100


def test_statistics_start_after_data_clear_error(db, hc):
    from home_thermal.hass import load_statistics
    with pytest.raises(ValueError):                         # observed: IndexError
        load_statistics(db, hc, start="2027-01-01")


# ---------------- 7. config validation gaps ----------------
@pytest.mark.parametrize("section", ["envelope", "internal", "emitters", "controls", "heat_pump",
                                     "boundaries", "solar"])
def test_null_section_gives_config_error_or_default(raw, section):
    try:
        cfg_with(raw, lambda c: c.update({section: None}))
    except ConfigError:
        pass                                                # observed: AttributeError / TypeError


@pytest.mark.parametrize("mut", [
    lambda c: c["zones"]["living"].update(ach="0.6"),
    lambda c: c["zones"]["living"].update(ach=-5),           # accepted: H_house becomes negative
    lambda c: c.update(internal=[["living", "hall", "30"]]),
    lambda c: c.update(internal=[["living", "hall", -30]]),  # accepted silently
    lambda c: c.update(parameters={"m_env": {"sd": 1}}),     # TypeError from dataclasses.replace
    lambda c: c.update(parameters={"m_env": {"bounds": [0.5]}}),
], ids=["ach_str", "ach_neg", "UA_str", "UA_neg", "param_field", "bounds_len1"])
def test_bad_values_give_config_error(raw, mut):
    with pytest.raises((ConfigError, KeyError)):
        c = cfg_with(raw, mut)
        parameter_set(c)
        Network(c)


def test_unknown_top_level_key_rejected(raw):
    with pytest.raises(ConfigError):                        # typo 'envelop' silently ignored
        cfg_with(raw, lambda c: c.update(envelop=c.pop("envelope")))


# ---------------- 8. CLI crashes with tracebacks instead of exit 2 ----------------
@pytest.mark.parametrize("case", ["bad_yaml", "warmup_too_long", "missing_db"])
def test_cli_errors_exit_2(tmp_path, case):
    from home_thermal.cli.main import main
    d = tmp_path
    main(["demo", str(d / "demo"), "--days", "4"])
    (d / "nullenv.yaml").write_text("zones:\n  a: {area: 1, volume: 2, sensor: T}\nenvelope:\n")
    (d / "bad.yaml").write_text("zones: [\n")
    (d / "h.yaml").write_text("entities: {T_out: sensor.x}\n")
    argv = {"null_envelope": ["check", str(d / "nullenv.yaml")],
            "bad_yaml": ["check", str(d / "bad.yaml")],
            "warmup_too_long": ["flows", str(d / "demo" / "house.yaml"), str(d / "demo" / "data.csv"),
                                "--warmup", "1000h"],
            "missing_db": ["hass-load", str(d / "h.yaml"), "--db", str(d / "nope.db"), "--out", str(d / "o.csv")]}
    assert main(argv[case]) == 2


def test_cli_accepts_empty_section(tmp_path):
    from home_thermal.cli.main import main
    (tmp_path / "h.yaml").write_text("zones:\n  a: {area: 1, volume: 2, sensor: T}\nenvelope:\n")
    assert main(["check", str(tmp_path / "h.yaml")]) == 0


# ---------------- 9. silent behaviour ----------------
def test_controller_missing_setpoint_column_errors():
    from home_thermal.control import Controller
    cfg = demo_house()
    df = synthetic_data(cfg, days=2).drop(columns=["SP_bedroom"])
    net = Network(cfg)
    tun = {z: {"Kp": 100, "Ti_h": 4} for z in ("living", "bedroom")}
    with pytest.raises((KeyError, ValueError)):             # observed: valve silently 0 % all run
        ctl = Controller(net, df, tun)
        simulate(net, df, controller=ctl)


def test_controller_rejects_invalid_pump_mode():
    from home_thermal.control import Controller
    cfg = demo_house()
    df = synthetic_data(cfg, days=1)
    with pytest.raises(ValueError, match="sometimes"):
        Controller(Network(cfg), df, {z: {"Kp": 100, "Ti_h": 4} for z in ("living", "bedroom")},
                   pump_modes={"P_living": "sometimes"})


def test_fit_on_data_shorter_than_warmup_errors():
    from home_thermal.calibrate import fit
    cfg = demo_house()
    df = synthetic_data(cfg, {"m_env": 2.0}, days=0.5)
    with pytest.raises(ValueError):                         # observed: silently returns the priors
        fit(cfg, df, free=["m_env"])


def test_simulate_with_first_row_sensors_missing():
    cfg = demo_house()
    df = synthetic_data(cfg, days=1)
    df.loc[df.index[0], ["T_living", "T_bedroom"]] = np.nan  # docs: NaN allowed in sensors
    simulate(Network(cfg), df)                               # observed: ValueError from initial_state


# ---------------- verification round ----------------
@pytest.mark.parametrize("mut", [
    lambda c: c["emitters"]["radiators"].update(bedroom=None),
    lambda c: c["controls"]["pumps"].update(P_living=None),
    lambda c: c["zones"]["living"].update(ach=float("nan")),
    lambda c: c["zones"]["living"].update(occupied="no"),
    lambda c: c["zones"]["living"].update(sensor=5),
    lambda c: c.update(parameters={"m_env": {"prior": 0}}),
    lambda c: c.update(parameters={"m_env": {"bounds": [-2, -1]}}),
    lambda c: c.update(parameters={"Q_int": {"prior": 5000}}),
], ids=["radiator_null", "pump_null", "nan", "bool_str", "sensor_num", "log_prior0", "log_bounds_neg", "prior_out"])
def test_more_bad_values_give_config_error(raw, mut):
    with pytest.raises(ConfigError):
        cfg_with(raw, mut)


def test_energy_balance_accepts_naive_start():
    cfg = demo_house()
    df = synthetic_data(cfg, days=2)
    net = Network(cfg)
    eb = energy_balance(net, simulate_detailed(net, df), start=pd.Timestamp("2026-01-06"))
    assert eb["hours"] == pytest.approx(24, abs=0.5)
