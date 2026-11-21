"""home_thermal.calibrate: fit a house model to measured data, validate it, and forecast.

    from home_thermal.calibrate import fit, cross_validate
    result = fit(cfg, df)                      # MAP fit of the default free parameters
    cv = cross_validate(cfg, df)               # blocked cross-validation (honest accuracy)
    print(cv.report(cfg))
"""
from .crossval import CrossValidation, cross_validate, make_blocks
from .estimate import FilterNoise, forecast_errors, kalman
from .fit import FitResult, FitSettings, data_terms, fit, meter_heat_true, residuals, score
from .io import load_params, save_params

__all__ = ["CrossValidation", "FilterNoise", "FitResult", "FitSettings", "cross_validate", "data_terms", "fit",
           "forecast_errors", "kalman", "load_params", "make_blocks", "meter_heat_true", "residuals",
           "save_params", "score"]
