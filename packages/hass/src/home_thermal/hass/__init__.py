"""home_thermal.hass: model input data from a Home Assistant recorder database.

    from home_thermal.hass import HassConfig, fetch_db, load_full
    hc = HassConfig.load("homeassistant.yaml")
    db = fetch_db(hc, "data/ha_db")          # optional: copy the DB over ssh (read-only)
    df = load_full(db, hc)                   # DataFrame in the home_thermal.core.data format
"""
from .source import (
                     HassConfig,
                     correct_water_temps,
                     fetch_db,
                     household_power,
                     load_full,
                     load_setpoints,
                     load_states,
                     load_statistics,
)

__all__ = ["HassConfig", "correct_water_temps", "fetch_db", "household_power", "load_full", "load_setpoints",
           "load_states", "load_statistics"]
