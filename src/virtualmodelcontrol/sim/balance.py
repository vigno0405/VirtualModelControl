"""The energy balance of a logged run: where the controller's energy went, and what it was given."""

from __future__ import annotations

import numpy as np

from .runlog import RunLog

NEEDS = ("energy/stored", "energy/kinetic", "power/port", "power/dissipation", "power/source")


def energy_balance(log: RunLog) -> dict[str, np.ndarray]:
    """The controller's energy over a run recorded with ``record="energy"``, step by step [J].

    ``energy`` is its stored plus kinetic energy; ``given`` the work it did on the robot through
    its port, ``dissipated`` what its dampers took and ``supplied`` what its sources gave, each
    integrated from the start. They give

        energy(t) − energy(0) = −given − dissipated + supplied + injected,

    where ``injected`` is whatever is left: the energy that changes of live Params put in, which
    ``controller.set`` returns as a jump, plus the error of the steps. ``margin`` is the energy
    the controller can still give: ``energy(0) + supplied − given``, never below 0 for a
    controller that is passive.
    """
    rows = log.arrays()
    missing = [name for name in NEEDS if name not in rows]
    if missing:
        raise ValueError(f"the log has no {missing}: run it with record=['energy']")
    t = rows["t"].ravel()

    def integral(power: np.ndarray) -> np.ndarray:
        """From the start to every step, by the trapezoid rule; a missing power counts as 0."""
        power = np.nan_to_num(power.ravel())
        steps = (power[1:] + power[:-1]) / 2 * np.diff(t)
        return np.concatenate([[0.0], np.cumsum(steps)])

    energy = rows["energy/stored"].ravel() + rows["energy/kinetic"].ravel()
    given = integral(rows["power/port"])
    dissipated = -integral(rows["power/dissipation"])
    supplied = integral(rows["power/source"])
    return {
        "t": t,
        "energy": energy,
        "given": given,
        "dissipated": dissipated,
        "supplied": supplied,
        "injected": energy - energy[0] + given + dissipated - supplied,
        "margin": energy[0] + supplied - given,
    }
