"""Mechanism: coordinates plus components, for a physical robot and a virtual one alike."""

from __future__ import annotations

from typing import Any

from numpy.typing import ArrayLike

from ..core.params import Param, ParamSet
from .components.base import Component
from .coordinates.base import walk
from .coordinates.frames import FramePoint
from .coordinates.joints import Joint, State


class Mechanism:
    """Named components acting on coordinates.

    A robot mechanism has a kinematic ``model`` (and optionally an ``actuation``); its components
    are physical (masses, identified stiffness). A controller is a mechanism without a model:
    its components act on the robot's coordinates and on its own virtual states.
    """

    def __init__(self, name: str, model: Any = None, actuation: Any = None) -> None:
        self.name = name
        self.model = model
        self.actuation = actuation
        self.components: dict[str, Component] = {}
        self.states: dict[str, State] = {}
        self._own_params: dict[str, Param] = {}

    def add(self, name: str, component: Component) -> Component:
        """Add ``component`` under ``name`` and return it."""
        if name in self.components:
            raise ValueError(f"{self.name!r} already has a component named {name!r}")
        self.components[name] = component
        return component

    def add_state(self, name: str, dim: int = 1, unit: str = "", initial: ArrayLike = 0.0) -> State:
        """Add a virtual degree of freedom (it needs an inertance) and return its coordinate."""
        if name in self.states:
            raise ValueError(f"{self.name!r} already has a state named {name!r}")
        state = State(name, dim, unit, initial)
        self.states[name] = state
        return state

    def add_param(self, param: Param) -> Param:
        """Add a Param that belongs to the mechanism itself (gravity, say) and return it."""
        self._own_params[param.name] = param
        return param

    def point(self, at: str | None = None, *, s: Any = None, offset: Any = None) -> FramePoint:
        """A point of the model: a named site, or arc parameter ``s`` [0, 1]."""
        if self.model is None:
            raise ValueError(f"{self.name!r} has no kinematic model")
        return FramePoint(self.model, at, s=s, offset=offset)

    def joint(self, index: Any) -> Joint:
        """Entries of the robot's generalized coordinates."""
        return Joint(index, getattr(self.model, "q_unit", ""))

    @property
    def space(self) -> Any:
        """Configuration space of the model."""
        if self.model is None:
            raise ValueError(f"{self.name!r} has no kinematic model")
        return self.model.space

    @property
    def params(self) -> ParamSet:
        """Every Param of the mechanism, by local name (component Params as ``name.param``)."""
        ps = ParamSet()
        if self.model is not None:
            ps.merge(self.model.params)
        if self.actuation is not None:
            ps.merge(self.actuation.params)
        for param in self._own_params.values():
            ps.add(param)
        for cname, component in self.components.items():
            for local, param in component.params().items():
                ps.add(param, f"{cname}.{local}", rename=True)
            for coord in walk(component.coord):
                for local, param in coord.params().items():
                    ps.add(param, f"{cname}.{local}", rename=True)
        return ps

    def __repr__(self) -> str:
        return f"Mechanism({self.name!r}, components={list(self.components)})"
