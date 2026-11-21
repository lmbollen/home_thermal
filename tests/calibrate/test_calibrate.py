import numpy as np
import pytest

from home_thermal.calibrate import cross_validate, fit, forecast_errors, kalman, load_params, save_params, score
from home_thermal.core import Network, parameter_set

from ..conftest import TRUTH


@pytest.fixture(scope="module")
def fitted(cfg, data):
    return fit(cfg, data)


def test_fit_beats_priors_and_recovers_heat_loss(cfg, data, fitted):
    rmse_fit = score(cfg, fitted.params, data).loc[["living", "bedroom"], "rmse"].mean()
    rmse_prior = score(cfg, {}, data).loc[["living", "bedroom"], "rmse"].mean()
    assert rmse_fit < 0.5 * rmse_prior
    assert rmse_fit < 0.15                          # sensor noise is 0.05 K
    h_true = Network(cfg, TRUTH).H_house()
    assert Network(cfg, fitted.params).H_house() == pytest.approx(h_true, rel=0.1)


def test_fit_respects_bounds_and_fixed(cfg, fitted):
    ps = parameter_set(cfg)
    for n in fitted.free:
        lo, hi = ps[n].bounds or (-np.inf, np.inf)
        assert lo <= fitted.params[n] <= hi
    assert fitted.params["m_win"] == 1.0            # fixed by default


def test_fit_without_heat_data_skips_heat_pump_parameters(cfg, data):
    res = fit(cfg, data.drop(columns=["HP_heat_W", "HP_elec_W"]), free=["m_env", "eta_hp"])
    assert res.free == ["m_env"]


def test_params_round_trip(cfg, fitted, tmp_path):
    save_params(tmp_path / "p.json", fitted.params, meta={"note": "test"})
    assert load_params(tmp_path / "p.json", cfg) == pytest.approx(fitted.params)


def test_kalman_and_forecasts(cfg, data, fitted):
    net = Network(cfg, fitted.params)
    XS, DS = kalman(net, data)
    assert DS.abs().iloc[200:].mean().max() < 100     # little unexplained heat in clean data
    origins = data.index[200::48][:4]
    fc = forecast_errors(net, data, origins, horizons_h=(1, 6), filt=(XS, DS))
    rmse = fc.groupby("method").error.apply(lambda e: np.sqrt((e ** 2).mean()))
    assert set(rmse.index) == {"twin+filter", "twin+filter, d=0", "persistence"}
    assert rmse["twin+filter, d=0"] < rmse["persistence"]


def test_cross_validation_runs(cfg, data):
    cv = cross_validate(cfg, data, free=["m_env", "m_vent", "Q_int", "m_rad"], block="72h", warmup="24h",
                        forecasts=False, progress=None)
    r = cv.room_rmse()
    assert len(r) == len(cv.blocks) >= 2
    assert (r["fitted"] < r["prior"]).all()
    assert "PARAMETERS across folds" in cv.report(cfg)
