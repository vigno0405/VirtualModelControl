"""Unit strings for parameters (labels only; every value inside the library is SI)."""

from __future__ import annotations

M = "m"
RAD = "rad"
S = "s"
KG = "kg"
N = "N"
NM = "N*m"
M_S2 = "m/s^2"
KG_M2 = "kg*m^2"


def force_unit(coord_unit: str) -> str:
    """Unit of the generalized force conjugate to a coordinate of unit ``coord_unit``."""
    return {M: N, RAD: NM}.get(coord_unit, f"J/{coord_unit}" if coord_unit else "J")


def stiffness_unit(coord_unit: str) -> str:
    """Force per coordinate unit."""
    return {M: "N/m", RAD: "N*m/rad"}.get(coord_unit, f"J/{coord_unit}^2" if coord_unit else "J")


def damping_unit(coord_unit: str) -> str:
    """Force per coordinate velocity."""
    return {M: "N*s/m", RAD: "N*m*s/rad"}.get(
        coord_unit, f"J*s/{coord_unit}^2" if coord_unit else "J*s"
    )


def inertance_unit(coord_unit: str) -> str:
    """Force per coordinate acceleration."""
    return {M: KG, RAD: KG_M2}.get(coord_unit, f"J*s^2/{coord_unit}^2" if coord_unit else "J*s^2")
