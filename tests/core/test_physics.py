import numpy as np
import pandas as pd
import yaml

from home_thermal.core import Network, normalise, radiator_heat
from home_thermal.core.synthetic import demo_house, demo_house_yaml


def heat(net, valve, t_flow=45.0, t_air=20.0):
    r = net.cfg["emitters"]["radiators"]["bedroom"]
    row = {"T_flow": t_flow, "T_water": t_flow - 3, "circ": 1.0}
    return radiator_heat(net, r, valve, row, t_air)


def test_ntu_radiator_behaviour(cfg):
    net = Network(cfg, {"v0": 20.0})
    assert heat(net, 10) == 0.0                      # below the dead band
    outs = [heat(net, v) for v in (30, 50, 75, 100)]
    assert all(np.diff(outs) > 0)                    # opens -> more heat
    assert heat(net, 100, t_flow=55) > heat(net, 100, t_flow=45)
    assert heat(net, 100, t_flow=18) == 0.0          # water colder than the room
    # never more than the catalogue surface limit at that temperature difference
    r = cfg["emitters"]["radiators"]["bedroom"]
    assert heat(net, 100) < r["q50"] * (25 / 49.83) ** r["n"] * 1.01


def test_catalogue_radiator_model():
    raw = yaml.safe_load(demo_house_yaml())
    raw["emitters"]["radiator_model"] = "catalogue"
    net = Network(normalise(raw))
    assert 0 < heat(net, 50) < heat(net, 100)


def test_vectorised_matches_scalar(cfg):
    net = Network(cfg)
    r = cfg["emitters"]["radiators"]["bedroom"]
    df = pd.DataFrame({"T_flow": [40.0, 45.0], "T_water": [37.0, 42.0], "circ": [1.0, 0.5]})
    v = np.array([60.0, 90.0])
    t_air = np.array([20.0, 19.0])
    vec = radiator_heat(net, r, v, df, t_air)
    sc = [radiator_heat(net, r, v[i], df.iloc[i].to_dict(), t_air[i]) for i in range(2)]
    assert np.allclose(vec, sc)


def test_demo_has_no_tank():
    assert Network(demo_house()).iT is None
