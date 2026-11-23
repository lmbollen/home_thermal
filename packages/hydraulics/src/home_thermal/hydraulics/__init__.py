"""home_thermal.hydraulics: compose pumps, pipes, valves, mixing valves, a bypass, a buffer and
emitters into a heating circuit, and solve its flows, pressures and water temperatures.

    from home_thermal.hydraulics import Circuit, Pump, PUMP_CURVES, Kv, HeatSource, Emitter
    c = Circuit(reference="return")
    c.add("hp", "return", "flow", [Pump(PUMP_CURVES["wilo-para-25-6"]), HeatSource(t_out=35)])
    c.add("rad", "flow", "return", [Kv(1.0), Emitter("T_room", q50=1500)])
    res = c.solve({"T_room": 20})
"""

from .circuit import Branch, Circuit, Result
from .components import (
    KPA_PER_M,
    PUMP_CURVES,
    Buffer,
    Element,
    Emitter,
    HeatSource,
    Kv,
    OverflowValve,
    Pipe,
    Pump,
    PumpCurve,
    ThermostaticMixingValve,
    ThrottlingLimiter,
    Valve,
    kv_dp,
    w_per_k,
)
from .spec import SpecError, build_circuit, load_spec, parameter_values

__all__ = [
    "KPA_PER_M",
    "PUMP_CURVES",
    "Branch",
    "Buffer",
    "Circuit",
    "Element",
    "Emitter",
    "HeatSource",
    "Kv",
    "OverflowValve",
    "Pipe",
    "Pump",
    "PumpCurve",
    "Result",
    "SpecError",
    "ThermostaticMixingValve",
    "ThrottlingLimiter",
    "Valve",
    "build_circuit",
    "kv_dp",
    "load_spec",
    "parameter_values",
    "w_per_k",
]
