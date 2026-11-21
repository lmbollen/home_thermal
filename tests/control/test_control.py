import pytest

from home_thermal.control import PI, Controller, fit_pi, replay_pi
from home_thermal.core import Network, sensor_temps, simulate


def test_pi_saturates_without_windup():
    c = PI(Kp=50, Ti_h=1)
    for _ in range(100):
        u = c.update(5.0, 0.25)                     # far too cold for a long time
    assert u == 100.0 and c.I <= 100.0
    # as soon as the room overshoots, the valve closes quickly (no wound-up integral)
    assert c.update(-1.0, 0.25) < 60


def test_fit_pi_recovers_tuning(cfg, data):
    df = data.copy()
    for z, t in cfg["controls"]["thermostats"].items():
        df[t["valve"]] = replay_pi(df[t["setpoint"]].to_numpy(), df[cfg["zones"][z]["sensor"]].to_numpy(),
                                   80.0, 3.0, 0.25)
    tuned = fit_pi(cfg, df)
    assert (tuned["rmse_train"] < 2.0).all()           # reproduces the recorded valves
    # the bedroom valve is often between 0 and 100 %, so its tuning is identifiable
    assert tuned.loc["bedroom", "Kp"] == pytest.approx(80.0, rel=0.1)
    assert tuned.loc["bedroom", "Ti_h"] == pytest.approx(3.0, rel=0.25)   # anti-windup makes Ti less sharp


def test_closed_loop_tracks_setpoint(cfg, data):
    net = Network(cfg)
    tuning = {z: {"Kp": 60.0, "Ti_h": 3.0} for z in cfg["controls"]["thermostats"]}
    df = data.copy()
    df["SP_living"] = 21.0
    df["SP_bedroom"] = 18.0
    ctl = Controller(net, df, tuning)
    X, Q = simulate(net, df, controller=ctl)
    S = sensor_temps(net, X, df).iloc[len(df) // 2:]
    assert abs(S["T_living"].mean() - 21.0) < 1.0      # floor heating lags: some overshoot
    assert abs(S["T_bedroom"].mean() - 18.0) < 0.5
    log = ctl.frame()
    assert {"V_living", "V_bedroom", "P_living"} <= set(log.columns)
    assert ((log["V_living"] > 0) == (log["P_living"] > 0)).all()     # pump follows the thermostat


def test_pump_mode_override(cfg, data):
    net = Network(cfg)
    tuning = {z: {"Kp": 60.0, "Ti_h": 3.0} for z in cfg["controls"]["thermostats"]}
    ctl = Controller(net, data, tuning, pump_modes={"P_living": "always"})
    simulate(net, data.iloc[:50], controller=ctl)
    assert (ctl.frame()["P_living"] == 1.0).all()
