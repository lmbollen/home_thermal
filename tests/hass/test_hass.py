import json
import sqlite3

import pandas as pd
import pytest

from home_thermal.hass import HassConfig, household_power, load_full, load_states

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
