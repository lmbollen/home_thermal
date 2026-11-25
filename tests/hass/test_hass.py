import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from home_thermal.calibrate import meter_heat_true
from home_thermal.hass import HassConfig, correct_defrost, household_power, load_full, load_states

T0 = pd.Timestamp("2026-11-10 00:00", tz="UTC").timestamp()


@pytest.fixture
def db(tmp_path):
    """A tiny recorder DB with the tables and columns home_thermal.hass reads."""
    path = tmp_path / "ha.db"
    con = sqlite3.connect(path)
    con.executescript("""
        create table states_meta (metadata_id integer primary key, entity_id text);
        create table state_attributes (attributes_id integer primary key, shared_attrs text);
        create table states (state_id integer primary key, metadata_id int, state text,
                             last_updated_ts real, attributes_id int);
        create table statistics_meta (id integer primary key, statistic_id text, unit_of_measurement text);
        create table statistics (id integer primary key, metadata_id int, start_ts real,
                                 mean real, min real, sum real);
    """)
    ents = ["sensor.t_room", "sensor.flow_t", "sensor.return_t", "sensor.flow", "switch.pump", "climate.room",
            "sensor.heat"]
    con.executemany("insert into states_meta values (?, ?)", list(enumerate(ents, 1)))
    con.execute("insert into state_attributes values (1, ?)", (json.dumps({"temperature": 20.5}),))
    rows = [(1, "19.0", 0), (1, "unavailable", 1800), (1, "19.6", 2400), (1, "19.6", 7200),
            (2, "40.85", 0), (3, "35.0", 0), (4, "0", 0), (4, "800", 600),
            (5, "off", 0), (5, "on", 900), (6, "heat", 0), (6, "off", 3000), (7, "5000", 0)]
    con.executemany("insert into states (metadata_id, state, last_updated_ts, attributes_id) values (?,?,?,?)",
                    [(m, s, T0 + t, 1 if m == 6 else None) for m, s, t in rows])
    con.commit()
    con.close()
    return path


@pytest.fixture
def hc():
    return HassConfig(entities={"T_room": "sensor.t_room", "T_flow": "sensor.flow_t", "T_return": "sensor.return_t",
                                "water_flow": "sensor.flow", "PUMP": "switch.pump", "HP_heat_W": "sensor.heat"},
                      setpoints={"SP_room": "climate.room"}, water_sensor_offset_K=0.85)


def test_states_are_step_held_and_averaged(db, hc):
    df = load_states(db, hc, step="15min")
    assert (df.index.to_series().diff().iloc[1:] == pd.Timedelta("15min")).all()
    assert df["T_room"].iloc[0] == pytest.approx(19.0)
    assert df["PUMP"].iloc[0] == 0.0
    assert df["PUMP"].iloc[1] == 1.0
    # 'unavailable' from minute 30 to 40 stays missing (not held): 30-45 min averages only 19.6
    assert df["T_room"].iloc[2] == pytest.approx(19.6)
    assert df["T_room"].iloc[3] == pytest.approx(19.6)


def test_water_offset_and_circulation(db, hc):
    df = load_states(db, hc, step="15min")
    assert df["T_flow"].iloc[0] == pytest.approx(40.85 - 0.425)
    assert df["T_return"].iloc[0] == pytest.approx(35.0 + 0.425)
    assert df["T_flow_raw"].iloc[0] == pytest.approx(40.85)
    assert df["circ"].iloc[0] == pytest.approx(5 / 15)        # flow > 100 L/h from minute 10
    assert df["HP_heat_per_K"].iloc[1] == pytest.approx(5000 / 5.85)


def test_setpoints(db, hc):
    df = load_states(db, hc, step="15min")
    assert df["SP_room"].iloc[0] == 20.5
    assert df["SP_room"].iloc[-1] == hc.setpoint_off      # thermostat switched off


def test_load_full_without_statistics(db, hc):
    hc.household_power = {"column": "P_app_W", "plus": ["HP_heat_W"], "minus": ["missing_col"]}
    df = load_full(db, hc)
    assert (df["P_app_W"] == df["HP_heat_W"].fillna(0)).all()


def test_household_power_clips():
    df = pd.DataFrame({"a": [100.0, 50.0], "b": [30.0, 80.0]})
    assert household_power(df, {"plus": ["a"], "minus": ["b"]}).tolist() == [70.0, 0.0]


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "ha.yaml"
    p.write_text("entities: {T_out: sensor.x}\nhots: oops\n")
    with pytest.raises(ValueError, match="hots"):
        HassConfig.load(p)


def test_offset_only_before_source_correction(db, hc):
    hc.water_offset_side = "flow"
    hc.water_offset_until = "2026-11-10 01:00"          # data from 01:00 on arrives corrected
    df = load_states(db, hc, step="15min")
    before, after = df.iloc[0], df.iloc[-1]
    assert before["T_flow"] == pytest.approx(40.85 - 0.85) and before["T_return"] == pytest.approx(35.0)
    assert after["T_flow"] == pytest.approx(40.85) and after["T_return"] == pytest.approx(35.0)
    assert after["HP_heat_per_K"] == 0.0                # heat already corrected: no dT_off correction
    assert before["HP_heat_per_K"] == 0.0 or df["HP_heat_per_K"].iloc[:4].max() > 0


def minutes(n=6, **cols):
    idx = pd.date_range("2026-11-24 00:00", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(cols, index=idx)


def spec(**kw):
    return {"W_per_K": 1000.0, "flag": None, "min_elec_W": 200.0, "min_dT": 1.0, "until": None, **kw}


def test_defrost_heat_from_the_water_cooling():
    # minutes 2-3: compressor on, water cooled 4 K, heat pump still reports +3 kW
    df = minutes(T_flow=[30, 30, 26, 26, 30, 30.0], T_return=[26, 26, 30, 30, 26, 26.0],
                 HP_heat_W=[4000, 4000, 3000, np.nan, 4000, 4000.0], HP_elec_W=[900, 900, 700, 700, 900, 50.0],
                 HP_heat_per_K=[1000, 1000, 0, 0, 1000, 1000.0])
    out = correct_defrost(df, spec())
    assert out["defrost"].tolist() == [0, 0, 1, 1, 0, 0]
    assert out["HP_heat_W"].tolist() == [4000, 4000, -4000, -4000, 4000, 4000]   # missing filled too
    assert out["HP_heat_per_K"].iloc[2:4].tolist() == [0, 0]


def test_defrost_flag_wins_where_present_detector_fills_gaps():
    df = minutes(T_flow=[26, 26, 26, 26, 30, 30.0], T_return=[30.0] * 4 + [26, 26.0],
                 HP_heat_W=[3000.0] * 6, HP_elec_W=[700.0] * 6, DF=[np.nan, np.nan, 0, 1, 1, 0])
    out = correct_defrost(df, spec(flag="DF"))
    # no flag yet: detected; flag 0 although cooling (e.g. after a stop): not defrost; flag 1: defrost
    assert out["defrost"].tolist() == [1, 1, 0, 1, 1, 0]
    assert out["HP_heat_W"].iloc[4] == pytest.approx(4000.0)     # flagged while warming: +4 K x 1000


def test_defrost_only_before_the_source_fixes_it():
    df = minutes(T_flow=[26.0] * 6, T_return=[30.0] * 6, HP_heat_W=[-5000.0] * 3 + [3000.0] * 3,
                 HP_elec_W=[700.0] * 6)
    out = correct_defrost(df, spec(until="2026-11-24 00:03"))
    assert out["defrost"].tolist() == [1] * 6
    assert out["HP_heat_W"].tolist() == [-4000.0] * 3 + [3000.0] * 3


def test_defrost_config_is_checked(tmp_path):
    p = tmp_path / "ha.yaml"
    p.write_text("entities: {T_out: sensor.x}\ndefrost: {W_per_K: 1330, flagg: df}\n")
    with pytest.raises(ValueError, match="flagg"):
        HassConfig.load(p)
    p.write_text("entities: {T_out: sensor.x}\ndefrost: {flag: df}\n")
    with pytest.raises(ValueError, match="W_per_K"):
        HassConfig.load(p)
    p.write_text("entities: {T_out: sensor.x}\ndefrost: {W_per_K: 1330}\n")
    assert HassConfig.load(p).defrost["min_dT"] == 1.0


def test_loader_applies_defrost(db, hc):
    hc.defrost = {"W_per_K": 1000.0, "min_elec_W": 0.0001}
    df = load_states(db, hc, step="15min")          # flow warmer than return throughout: no defrost
    assert (df["defrost"] == 0).all()


def test_meter_keeps_negative_heat():
    df = pd.DataFrame({"HP_heat_W": [-4000.0, 100.0, 5000.0], "HP_heat_per_K": [0.0, 200.0, 1000.0]})
    q = meter_heat_true(df, {"k_meter": 0.9, "dT_off": 0.85})
    assert q.tolist() == pytest.approx([-3600.0, 0.0, 0.9 * (5000 - 850)])
