"""Zone-level RC network (grey-box thermal model).

Nodes per zone: an air node (air + furniture) and a mass node (walls/floors, ISO 13790 style).
Zones with floor heating get a slab node between water, air and ground; a hot-water tank (if any)
is one more node. Boundaries: outdoor air (0), ground (1), neighbour (2).

    C dx/dt = -G x + Gb T_b + q

The linear part is discretised exactly (matrix exponential). Heat inputs q (emitters, sun,
background gain) are evaluated per step in `physics.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from .params import check_names, defaults

RHO_CP_AIR = 1200.0   # J/m3K
BOUNDARIES = ("outdoor", "ground", "neighbour")
LOSS_PARTS = ("opaque", "windows", "vent", "ground", "party")


@dataclass
class Link:
    """One heat path: node i to node j, or node i to boundary b. g in W/K."""
    i: int
    j: int | None
    b: int | None
    g: float
    category: str
    zone: str | tuple


@dataclass
class Network:
    """The RC network of a house for one parameter set.

    cfg: normalised house configuration (`config.load_house`); p: parameter values. Parameters
    missing from p are taken at their prior (`params.defaults`); unknown names raise KeyError.
    Build a new Network for new parameters (matrices are computed once, here)."""
    cfg: dict
    p: dict = field(default_factory=dict)
    zones: list = field(init=False)
    names: list = field(init=False)

    def __post_init__(self):
        check_names(self.cfg, self.p)
        self.p = {**defaults(self.cfg), **self.p}
        cfg, p = self.cfg, self.p
        self.zones = list(cfg["zones"])
        self.ufh_zones = list(cfg["emitters"]["floor_heating"])
        nz = len(self.zones)
        self.iA = {z: i for i, z in enumerate(self.zones)}
        self.iM = {z: nz + i for i, z in enumerate(self.zones)}
        self.iS = {z: 2 * nz + i for i, z in enumerate(self.ufh_zones)}
        n = 2 * nz + len(self.ufh_zones)
        self.iT = n if cfg["dhw"] else None
        self.n = n = n + (1 if cfg["dhw"] else 0)
        self.names = ([f"air:{z}" for z in self.zones] + [f"mass:{z}" for z in self.zones]
                      + [f"slab:{z}" for z in self.ufh_zones] + (["tank:dhw"] if cfg["dhw"] else []))
        self.G, self.Gb, self.C = np.zeros((n, n)), np.zeros((n, len(BOUNDARIES))), np.zeros(n)
        self.links: list[Link] = []
        self.H_zone = {z: dict.fromkeys(LOSS_PARTS, 0.0) for z in self.zones}
        self.solar_share = np.zeros(n)
        tm = cfg["thermal_mass"]
        for z, zc in cfg["zones"].items():
            a, m = self.iA[z], self.iM[z]
            self.C[a] = zc["volume"] * RHO_CP_AIR * tm["air_furniture_factor"] * p["m_Cair"]
            self.C[m] = zc["area"] * tm["C_m_per_m2"] * p["m_Cm"]
            self._link(a, m, zc["area"] * tm["H_am_per_m2"] * p["m_Ham"], "air-mass", z)
            hv = zc["ach"] * zc["volume"] * RHO_CP_AIR / 3600 * p["m_vent"]
            self._bound(a, 0, hv, "ventilation", z, "vent")
            for el in cfg["envelope"].get(z, []):
                self._envelope_element(z, el)
        for z1, z2, ua, *kind in cfg["internal"]:
            m = p["m_air"] if (kind and kind[0] == "air") else p["m_int"]
            self._link(self.iA[z1], self.iA[z2], ua * m, "internal", (z1, z2))
        for z, u in cfg["emitters"]["floor_heating"].items():
            self.C[self.iS[z]] = u["area"] * u["c"]
            self._link(self.iS[z], self.iA[z], u["area"] * u["h_sa"], "slab-air", z)
        if cfg["dhw"]:
            self.C[self.iT] = cfg["dhw"]["volume_l"] * 4186.0
            self._link(self.iT, self.iA[cfg["dhw"]["zone"]], p["UA_tank"], "tank-loss", cfg["dhw"]["zone"])
        if self.solar_share.sum() > 0:
            self.solar_share /= self.solar_share.sum()
        occ = {z: zc["area"] for z, zc in cfg["zones"].items() if zc["occupied"]}
        self.int_share = np.zeros(n)
        for z, area in occ.items():
            self.int_share[self.iA[z]] = area / sum(occ.values())

    def _link(self, i, j, g, category, zone):
        G = self.G
        G[i, i] += g
        G[j, j] += g
        G[i, j] -= g
        G[j, i] -= g
        self.links.append(Link(i, j, None, g, category, zone))

    def _bound(self, i, b, g, category, zone, part):
        self.G[i, i] += g
        self.Gb[i, b] += g
        self.H_zone[zone][part] += g
        self.links.append(Link(i, None, b, g, category, zone))

    def _envelope_element(self, z, el):
        p, a = self.p, self.iA[z]
        ua, kind = el["area"] * el["U"], el["kind"]
        if kind in ("wall", "roof"):
            self._bound(a, 0, ua * p["m_env"], kind + "s", z, "opaque")
        elif kind in ("window", "door"):
            self._bound(a, 0, ua * p["m_win"], "windows+doors", z, "windows")
            if "solar" in el:
                self.solar_share[a] += el["area"] * self.cfg["solar"]["weights"][el["solar"]]
        elif kind == "floor":
            node = self.iS[z] if (el.get("ufh") and z in self.iS) else a
            self._bound(node, 1, ua * p["m_floor"], "ground", z, "ground")
        elif kind == "party":
            self._bound(a, 2, ua * p["m_party"], "neighbour", z, "party")

    def discretise(self, dt: float):
        """Exact zero-order-hold discretisation for step dt (s): x' = Ad x + Bb T_b + Bq q."""
        n = self.n
        Ci = 1.0 / self.C
        A = -self.G * Ci[:, None]
        B = np.hstack([self.Gb * Ci[:, None], np.diag(Ci)])
        M = np.zeros((n + B.shape[1], n + B.shape[1]))
        M[:n, :n] = A
        M[:n, n:] = B
        E = expm(M * dt)
        return E[:n, :n], E[:n, n:n + 3], E[:n, n + 3:]

    @property
    def H_parts(self) -> dict:
        """Boundary conductance per loss path, summed over all zones (W/K)."""
        return {k: sum(h[k] for h in self.H_zone.values()) for k in LOSS_PARTS}

    def H_house(self) -> float:
        """House heat-loss coefficient to outdoor air and ground (W/K). Excludes the party wall and
        zones with `in_house_loss: false` (e.g. a heated shed)."""
        return sum(v for z, d in self.H_zone.items() if self.cfg["zones"][z]["in_house_loss"]
                   for k, v in d.items() if k != "party")

    def boundary_temps(self, t_out: float) -> np.ndarray:
        """Boundary temperatures [outdoor, ground, neighbour]."""
        # without floor / party elements these temperatures are not parameters and are unused
        return np.array([t_out, self.p.get("T_ground", 10.0), self.p.get("T_neighbour", 18.0)])
