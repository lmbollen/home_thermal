"""home_thermal.core: a grey-box thermal model (RC network) of a house, built from a YAML config.

    from home_thermal.core import load_house, Network, simulate, read_csv
    cfg = load_house("house.yaml")
    net = Network(cfg)                 # parameters at their priors
    states, heat = simulate(net, read_csv("data.csv"))
"""
from .analysis import coupling, energy_balance, simulate_detailed
from .config import ConfigError, load_house, normalise
from .data import check_inputs, check_sensors, prepare, read_csv, required_columns, step_seconds
from .heatpump import HeatPump
from .network import Network
from .params import HEAT_PUMP_PARAMS, Param, check_names, complete, default_free, defaults, parameter_set
from .physics import (
                      emitter_heat,
                      free_gains,
                      radiator_heat,
                      sensor_lag,
                      sensor_temps,
                      valve_sensor_warming,
                      valve_value,
)
from .simulate import initial_state, simulate

__all__ = [
    "HEAT_PUMP_PARAMS", "ConfigError", "HeatPump", "Network", "Param", "check_inputs", "check_names",
    "check_sensors", "complete", "coupling", "default_free", "defaults", "emitter_heat", "energy_balance",
    "free_gains", "initial_state", "load_house", "normalise", "parameter_set", "prepare", "radiator_heat",
    "read_csv", "required_columns", "sensor_lag", "sensor_temps", "simulate", "simulate_detailed",
    "step_seconds", "valve_sensor_warming", "valve_value",
]
