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


@pytest.mark.bench
def test_soft_arm_plan_builds_in_seconds_and_solves_again_in_a_fraction():
    import time

    from soft_arm_plan import HORIZON, NODES, SCALES, TRANSITION, soft_arm_swap
    from virtualmodelcontrol import optimization as opt

    _, tip, goal, hold, new = soft_arm_swap()
    problem = opt.Problem(new)
    problem.add(
        opt.Collocation(
            np.zeros(9), HORIZON, NODES, initial=hold, transition=TRANSITION, scales=SCALES
        )
    )
    problem.free("new.pull.stiffness")
    problem.add(opt.Effort(1e-4))
    problem.add(opt.Cost(tip - goal, t_from=TRANSITION))
    start = time.perf_counter()
    first = problem.solve()  # builds the program and the solver, then solves
    built = time.perf_counter() - start
    start = time.perf_counter()
    again = problem.solve()
    solved = time.perf_counter() - start
    assert first.converged and again.converged
    assert built < 60.0 and solved < 10.0, (built, solved)
