"""A robot and the virtual mechanism that controls it, as one system."""

from __future__ import annotations

from typing import Any

from .core.params import ParamSet
from .mechanisms.components.base import Component
from .mechanisms.coordinates.joints import State
from .mechanisms.mechanism import Mechanism
from .models.actuation import Direct


class VirtualMechanismSystem:
    """A robot mechanism (with a kinematic model) and the virtual mechanism attached to it.

    Params are named ``<mechanism>.<name>``; a Param shared by the robot and the controller keeps
    the robot's name. ``couplings`` is an optional third mechanism, treated like the controller.
    """

    def __init__(
        self, robot: Mechanism, virtual: Mechanism, couplings: Mechanism | None = None
    ) -> None:
        if robot.model is None:
            raise ValueError(f"robot {robot.name!r} needs a kinematic model")
        self.robot = robot
        self.virtual = virtual
        self.couplings = couplings
        self.actuation: Any = robot.actuation if robot.actuation is not None else Direct()

    @property
    def controllers(self) -> list[Mechanism]:
        """The virtual mechanism and the couplings, if any."""
        return [m for m in (self.virtual, self.couplings) if m is not None]

    @property
    def params(self) -> ParamSet:
        """Every Param of the system, robot first (with the default actuation's, if it has none)."""
        ps = ParamSet()
        for mechanism in (self.robot, *self.controllers):
            ps.merge(mechanism.params, mechanism.name)
        if self.robot.actuation is None:
            ps.merge(self.actuation.params, self.robot.name)
        return ps

    @property
    def states(self) -> list[State]:
        """Virtual degrees of freedom, in order."""
        return [s for m in self.controllers for s in m.states.values()]

    @property
    def components(self) -> list[tuple[str, Component]]:
        """Controller components as (``mechanism.name``, component)."""
        return [(f"{m.name}.{n}", c) for m in self.controllers for n, c in m.components.items()]

    def __repr__(self) -> str:
        return f"VirtualMechanismSystem({self.robot.name!r}, {self.virtual.name!r})"
