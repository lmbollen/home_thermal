"""home_thermal.control: thermostats and pump logic for closed-loop what-if simulations.

    from home_thermal.control import Controller, fit_pi
    tuning = fit_pi(cfg, df)                              # Kp/Ti per thermostat group
    ctl = Controller(net, df, tuning, pump_modes={"UFH_A": "always"})
    states, heat = simulate(net, df, controller=ctl)
"""
from .controllers import PI, Controller, fit_pi, replay_pi

__all__ = ["PI", "Controller", "fit_pi", "replay_pi"]
