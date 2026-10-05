"""Replays: a logged run's commands sent again to a simulated robot, to compare with the run."""

from __future__ import annotations

from typing import Any

from ..core.signals import Signals
from .runlog import RunLog


def replay(log: RunLog, plant: Any) -> RunLog:
    """The logged motor torques sent again to the simulated ``plant``, each for as long as it was
    held in the run; returns the simulated run, on the logged times.

    The plant starts where the run started: at its logged ``q`` and ``v``, or else at its first
    motor angles and rates (``plant.state_from_motors``). ``compare(log, replayed)`` then shows how
    far the model is from the robot.
    """
    rows = log.arrays()
    t, torques = rows["t"].ravel(), rows["motor_torque"]
    if "q" in rows and "v" in rows:
        start = (rows["q"][0], rows["v"][0])
    else:
        start = plant.state_from_motors(rows["motor_position"][0], rows["motor_velocity"][0])
    plant.reset(start)
    out = RunLog(meta={"replay of": log.meta.get("start", "")})
    for k, (time, torque) in enumerate(zip(t, torques, strict=True)):
        meas = plant.read()
        plant.write(Signals(plant.t, motor_torque=torque))
        out.step(t=time, motor_torque=torque, **{name: meas[name] for name in meas.names})
        if k + 1 < len(t):
            plant.advance(t[k + 1] - time)
    return out
