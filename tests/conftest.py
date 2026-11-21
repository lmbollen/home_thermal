import pytest

from home_thermal.core.synthetic import demo_house, synthetic_data

TRUTH = {"m_env": 1.3, "m_vent": 0.8, "Q_int": 250.0, "m_rad": 1.2}


@pytest.fixture(scope="session")
def cfg():
    return demo_house()


@pytest.fixture(scope="session")
def data(cfg):
    return synthetic_data(cfg, TRUTH, days=8, seed=1)
