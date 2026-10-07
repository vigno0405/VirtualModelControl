"""The equations of motion of a robot from a function, for models that are not written as parts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Equations:
    """Equations of motion from a function, for a black-box, learned or external model.

    ``residual(q, v, a, tau, f, p)`` returns the vector r = M(q) a + h(q, v) − tau − f, which is
    zero along a motion: it must be affine in ``a``, written with CasADi operations. ``tau`` is
    the generalized force of the motors (through the robot's actuation) and ``f`` that of the
    robot's own components (its springs, dampers, gravity, contact); ``p`` maps the model's Param
    names to their values, as for ``frame``. The robot then has no inertances: the masses are
    in the equations.

    ``energy(q, v, p)``, when there is one, returns the kinetic and the stored energy (T, V) of
    what the residual describes; the components' storage adds to V. The passivity tools need
    the energy and refuse a robot without it: they read the work of the motors and what the
    dampers take, which the equations alone do not give.
    """

    def __init__(
        self,
        residual: Callable[[Any, Any, Any, Any, Any, dict[str, Any]], Any],
        energy: Callable[[Any, Any, dict[str, Any]], tuple[Any, Any]] | None = None,
    ) -> None:
        self.residual, self.energy = residual, energy
