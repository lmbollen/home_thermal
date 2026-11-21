import json

from home_thermal.cli import main

FIX = "m_Cm,m_Ham,T_ground,T_neighbour,m_floor,m_party,g_sol,m_ufh,v0,eta_hp,P_aux_hp,k_meter"


def test_demo_check_fit_simulate(tmp_path, capsys):
    out = tmp_path / "demo"
    assert main(["demo", str(out), "--days", "5"]) == 0
    assert main(["check", str(out / "house.yaml")]) == 0
    assert "required data columns" in capsys.readouterr().out
    p = tmp_path / "p.json"
    assert main(["fit", str(out / "house.yaml"), str(out / "data.csv"), "--out", str(p), "--fix", FIX]) == 0
    assert "m_env" in json.loads(p.read_text())["params"]
    assert main(["simulate", str(out / "house.yaml"), str(out / "data.csv"), "--params", str(p),
                 "--out", str(tmp_path / "sim.csv")]) == 0
    assert main(["flows", str(out / "house.yaml"), str(out / "data.csv")]) == 0


def test_bad_config_gives_clean_error(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("zones: {a: {area: 0, volume: 10}}\n")
    assert main(["check", str(bad)]) == 2
    assert "zones.a.area" in capsys.readouterr().err
