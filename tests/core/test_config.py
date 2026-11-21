import copy

import pytest
import yaml

from home_thermal.core import ConfigError, default_free, defaults, normalise, parameter_set
from home_thermal.core.synthetic import demo_house, demo_house_yaml


def raw():
    return yaml.safe_load(demo_house_yaml())


def test_demo_normalises_with_defaults(cfg):
    assert cfg["zones"]["hall"]["sensor"] is None
    assert cfg["zones"]["living"]["occupied"] is True
    assert cfg["emitters"]["radiators"]["bedroom"]["sensor_on_valve"] is False
    assert cfg["controls"]["thermostats"]["living"]["group"] == "living"


def test_normalise_does_not_modify_input():
    r = raw()
    before = copy.deepcopy(r)
    normalise(r)
    assert r == before


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["envelope"]["living"].append({"kind": "wal", "area": 1, "U": 1}), "kind"),
    (lambda c: c["internal"].append(["living", "kitchen", 10]), "unknown zone 'kitchen'"),
    (lambda c: c["zones"]["living"].update(area=-3), "zones.living.area"),
    (lambda c: c["emitters"]["radiators"]["bedroom"].pop("valve"), "valve"),
    (lambda c: c["emitters"]["radiators"].update(hall={"q50": 500, "valve": "V", "sensor_on_valve": True}),
     "has no sensor"),
    (lambda c: c["envelope"]["living"].append({"kind": "window", "area": 1, "U": 1, "solar": "UP"}), "orientation"),
    (lambda c: [z.update(sensor=None) for z in c["zones"].values()], "at least one zone needs a 'sensor'"),
    (lambda c: c["controls"]["pumps"]["P_living"].update(mode="sometimes"), "mode"),
])
def test_invalid_configs_name_the_problem(mutate, message):
    r = raw()
    mutate(r)
    with pytest.raises(ConfigError, match=message):
        normalise(r)


def test_parameters_follow_the_house(cfg):
    ps = parameter_set(cfg)
    assert "f_off_P_living" in ps and "m_rad" in ps and "v0" in ps
    assert "cop_dhw" not in ps            # no hot-water tank
    assert "kappa_q" not in ps            # no valve-mounted sensor
    assert "UA_hp" not in ps              # heat pump has no zone
    assert "m_win" not in default_free(cfg)


def test_parameter_overrides():
    r = raw()
    r["parameters"] = {"m_win": {"fixed": False, "sigma": 0.3}, "g_sol": {"prior": 3.0, "bounds": [1, 5]}}
    cfg = normalise(r)
    assert "m_win" in default_free(cfg)
    assert defaults(cfg)["g_sol"] == 3.0
    assert parameter_set(cfg)["g_sol"].bounds == (1, 5)
    r["parameters"] = {"cop_dhw": {"prior": 3.0}}
    with pytest.raises(ConfigError, match="cop_dhw"):
        normalise(r)


def test_demo_house_is_stable():
    assert demo_house() == demo_house()
