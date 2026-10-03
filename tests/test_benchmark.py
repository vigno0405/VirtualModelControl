"""Timing targets; opt-in with ``pytest -m bench`` (machines differ)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc

spec = importlib.util.spec_from_file_location(
    "tick", Path(__file__).parents[1] / "benchmarks" / "tick.py"
)
tick = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tick)


@pytest.mark.bench
def test_soft_arm_controller_step_under_30_us():
    law = tick.build()
    controller = vmc.VMCController(law)
    rng = np.random.default_rng(0)
    meas = [
        vmc.Signals(0.0, motor_position=rng.uniform(-0.3, 0.3, 9), motor_velocity=np.zeros(9))
        for _ in range(50)
    ]
    assert tick.per_tick(lambda m: controller.step(0.0, m), meas) < 30.0
