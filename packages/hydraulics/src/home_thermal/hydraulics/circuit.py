"""Compose components into a circuit and solve it.

A circuit is a graph: nodes (junctions, by name) and branches between them. A branch is a series
of elements (pipes, valves, pumps, emitters, a heat source, a buffer) or an overflow valve.
One node is the pressure reference (where the expansion vessel connects).

`solve(ctx)` finds, for the given inputs `ctx` (data columns and space temperatures, e.g. one row
of the model data), the flow in every branch and the pressure at every node, then the water
temperatures by mixing at the nodes. Thermostatic mixing valves couple the two: their opening
depends on the mixed temperature, so hydraulics and temperatures are iterated together.
`step(ctx, dt)` solves and then advances storage (buffers) over dt seconds.

    c = Circuit(reference="return")
    c.add("hp", "return", "flow", [Pump(curve), HeatSource(t_out=35)])
    c.add("load", "flow", "return", [Kv(1.5), Emitter("T_room", UA=150)])
    res = c.solve({"T_room": 20})
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, root

from .components import Buffer, Element, OverflowValve, ThermostaticMixingValve, w_per_k


@dataclass
class Branch:
    name: str
    a: str
    b: str
    elements: list = field(default_factory=list)  # series elements, or [] for an overflow branch
    overflow: OverflowValve | None = None

    def dp(self, q: float, ctx: dict) -> float:
        return sum(e.dp(q, ctx) for e in self.elements)


@dataclass
class Result:
    flow: dict  # branch -> m3/h (positive a -> b)
    pressure: dict  # node -> kPa relative to the reference node
    temperature: dict  # node -> C
    element_in: dict  # (branch, element index) -> inlet temperature C
    element_out: dict  # (branch, element index) -> outlet temperature C
    heat: dict  # element name -> W added to the water (negative: removed)
    iterations: int

    def dp(self, a: str, b: str) -> float:
        return self.pressure[a] - self.pressure[b]


class Circuit:
    def __init__(self, reference: str):
        self.reference = reference
        self.branches: dict[str, Branch] = {}
        self.mixing_valves: list[ThermostaticMixingValve] = []
        self._x = None
        self._t: dict[str, float] = {}

    # ------------------------------------------------------------- building
    def add(self, name: str, a: str, b: str, elements: list[Element]) -> Branch:
        """A branch from node a to node b through `elements` in series (flow a -> b positive)."""
        if name in self.branches:
            raise ValueError(f"branch {name!r} exists")
        if a == b:
            raise ValueError(f"branch {name!r} connects node {a!r} to itself")
        br = Branch(name, a, b, list(elements))
        self.branches[name] = br
        return br

    def add_overflow(self, name: str, a: str, b: str, valve: OverflowValve) -> Branch:
        br = self.add(name, a, b, [])
        br.overflow = valve
        return br

    def add_mixing_valve(
        self, tmv: ThermostaticMixingValve, hot_from: str, cold_from: str, mixed: str, hot_extra=(), cold_extra=()
    ) -> None:
        """A thermostatic 3-way valve: hot port from node `hot_from`, cold (recirculation) port from
        `cold_from`, mixed outlet at node `mixed`."""
        tmv.mixed_node = mixed
        self.add(f"{tmv.name}.hot", hot_from, mixed, [*hot_extra, tmv.hot])
        self.add(f"{tmv.name}.cold", cold_from, mixed, [*cold_extra, tmv.cold])
        self.mixing_valves.append(tmv)

    @property
    def nodes(self) -> list[str]:
        out = []
        for br in self.branches.values():
            for n in (br.a, br.b):
                if n not in out:
                    out.append(n)
        return out

    def elements(self):
        for br in self.branches.values():
            yield from ((br, i, e) for i, e in enumerate(br.elements))

    # ------------------------------------------------------------- hydraulics
    def _topology(self):
        key = tuple((nm, br.a, br.b) for nm, br in self.branches.items())
        if getattr(self, "_topo_key", None) != key:
            names, nodes = list(self.branches), self.nodes
            if self.reference not in nodes:
                raise ValueError(f"reference node {self.reference!r} is not connected")
            free = [n for n in nodes if n != self.reference]
            idx = {n: i for i, n in enumerate(free)}
            A = np.zeros((len(free), len(names)))  # node-branch incidence (+1 in, -1 out)
            for k, nm in enumerate(names):
                br = self.branches[nm]
                if br.a in idx:
                    A[idx[br.a], k] -= 1.0
                if br.b in idx:
                    A[idx[br.b], k] += 1.0
            self._topo = (names, free, idx, A)
            self._topo_key = key
        return self._topo

    def _hydraulics(self, ctx: dict):
        names, free, idx, A = self._topology()
        nb, nf = len(names), len(free)
        brs = [self.branches[nm] for nm in names]
        DP = -A.T  # branch pressure difference p_a - p_b = DP @ p (reference = 0)
        h = 1e-6

        def residual(x):
            q, p = x[:nb], x[nb:]
            dp = DP @ p
            r = np.empty(nb + nf)
            for k, br in enumerate(brs):
                r[k] = (
                    q[k] - br.overflow.flow(dp[k], ctx)
                    if br.overflow is not None
                    else (dp[k] - br.dp(q[k], ctx)) / 10.0
                )
            r[nb:] = A @ q
            return r

        def jacobian(x):
            q, p = x[:nb], x[nb:]
            dp = DP @ p
            J = np.zeros((nb + nf, nb + nf))
            for k, br in enumerate(brs):
                if br.overflow is not None:
                    d = (br.overflow.flow(dp[k] + 1e-4, ctx) - br.overflow.flow(dp[k] - 1e-4, ctx)) / 2e-4
                    J[k, k] = 1.0
                    J[k, nb:] = -d * DP[k]
                else:
                    J[k, k] = -(br.dp(q[k] + h, ctx) - br.dp(q[k] - h, ctx)) / (2 * h) / 10.0
                    J[k, nb:] = DP[k] / 10.0
            J[nb:, :nb] = A
            return J

        x0 = self._x if self._x is not None and len(self._x) == nb + nf else np.full(nb + nf, 0.1)
        sol = root(residual, x0, jac=jacobian, method="hybr")
        err = np.abs(residual(sol.x)).max()
        if err > 1e-6:
            sol2 = root(residual, np.full_like(x0, 0.1), jac=jacobian, method="lm")
            err2 = np.abs(residual(sol2.x)).max()
            if err2 < err:
                sol, err = sol2, err2
        if err > 1e-4:
            raise RuntimeError(f"hydraulic solve did not converge (max residual {err:.2g})")
        self._x = sol.x
        flow = dict(zip(names, sol.x[:nb]))
        pressure = {self.reference: 0.0, **{n: sol.x[nb + i] for n, i in idx.items()}}
        return flow, pressure

    # ------------------------------------------------------------- temperatures
    def _temperatures(self, flow: dict, ctx: dict, sweeps: int = 500):
        nodes = self.nodes
        t = {n: self._t.get(n, 30.0) for n in nodes}
        t_in, t_out = {}, {}
        for _ in range(sweeps):
            inflow = {n: [] for n in nodes}
            for nm, br in self.branches.items():
                q = flow[nm]
                if abs(q) < 1e-9:
                    continue
                src, dst = (br.a, br.b) if q > 0 else (br.b, br.a)
                temp = t[src]
                seq = list(enumerate(br.elements)) if q > 0 else list(reversed(list(enumerate(br.elements))))
                for i, e in seq:
                    t_in[(nm, i)] = temp
                    temp = e.outlet(temp, q, ctx)
                    t_out[(nm, i)] = temp
                inflow[dst].append((abs(q), temp))
            change = 0.0
            for n in nodes:
                if inflow[n]:
                    new = sum(q * tt for q, tt in inflow[n]) / sum(q for q, _ in inflow[n])
                    change = max(change, abs(new - t[n]))
                    t[n] = new
            if change < 1e-7:
                break
        self._t = t
        return t, t_in, t_out

    # ------------------------------------------------------------- public
    def solve(self, ctx: dict, passes: int = 6, tol: float = 1e-4) -> Result:
        """Flows, pressures and temperatures for the inputs in ctx.

        Each thermostatic mixing valve settles where its opening matches what its wax element wants
        at the resulting mixed temperature: u = target(T_mixed(u)). That function is monotone, so
        it is solved by bracketing (Brent) per valve; several valves are visited in turn."""

        def run():
            flow, pressure = self._hydraulics(ctx)
            t, t_in, t_out = self._temperatures(flow, ctx)
            return flow, pressure, t, t_in, t_out

        it = 0
        for it in range(1, passes + 1):  # noqa: B007  (it is reported in the Result)
            moved = 0.0
            for v in self.mixing_valves:
                old = v.u

                def g(u, v=v):
                    v.u = u
                    return u - v.target(run()[2][v.mixed_node])

                # bracket around the previous opening first (cheap when inputs change slowly)
                lo, hi, step = max(old - 0.05, 0.0), min(old + 0.05, 1.0), 0.05
                glo, ghi = g(lo), g(hi)
                while glo > 0 and lo > 0:
                    step *= 3
                    hi, ghi, lo = lo, glo, max(lo - step, 0.0)
                    glo = g(lo)
                while ghi < 0 and hi < 1:
                    step *= 3
                    lo, glo, hi = hi, ghi, min(hi + step, 1.0)
                    ghi = g(hi)
                if glo >= 0:
                    v.u = lo
                elif ghi <= 0:
                    v.u = hi
                else:
                    v.u = brentq(g, lo, hi, xtol=tol)
                moved = max(moved, abs(v.u - old))
            if moved < tol or len(self.mixing_valves) < 2:
                break
        flow, pressure, t, t_in, t_out = run()
        heat = {}
        for br, i, e in self.elements():
            if (br.name, i) in t_in:
                heat[e.name] = heat.get(e.name, 0.0) + w_per_k(flow[br.name]) * (
                    t_out[(br.name, i)] - t_in[(br.name, i)]
                )
        return Result(flow, pressure, t, t_in, t_out, heat, it)

    def step(self, ctx: dict, dt_s: float) -> Result:
        """Solve, then advance every buffer over dt_s with what flowed into it."""
        res = self.solve(ctx)
        for br, i, e in self.elements():
            if isinstance(e, Buffer) and (br.name, i) in res.element_in:
                e.advance([(res.flow[br.name], res.element_in[(br.name, i)])], dt_s, ctx)
        return res
