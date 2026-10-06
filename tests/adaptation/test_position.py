import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import PositionRegulation
from virtualmodelcontrol.robots import helyx

TARGET = np.array([0.08, 0.0, 0.40])


@pytest.fixture(scope="module")
def world():
    arm = helyx.add_dynamics(helyx.arm("145-145-145"))
    tip = arm.point(s=1.0)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(tip - vmc.Ref("goal", 3, value=TARGET), 300.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    return arm, vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))


def run(world, regulate, seconds=6.0, gain=0.02):
    arm, compiled = world
    controller = vmc.VMCController(compiled)
    law = PositionRegulation(controller, {"ctrl.reach.goal": 1.0}, rate=gain)
    plant = vmc.sim.ModelPlant(arm)
    kin = vmc.Kinematics(arm)
    dt = 1 / helyx.CONTROL_RATE
    errors = []
    for _ in range(int(seconds / dt)):
        plant.write(controller.step(plant.t, plant.read()))
        if regulate and plant.t > 2.0:
            law.step(controller, {"ctrl.reach.goal": TARGET})
        errors.append(np.linalg.norm(TARGET - kin.position(plant.q, 1.0)))
        plant.advance(dt)
    return np.array(errors), controller.live_params()["ctrl.reach.goal"]


def test_the_arms_own_stiffness_leaves_the_tip_short_of_the_goal(world):
    errors, goal = run(world, regulate=False)
    assert 0.01 < errors[-1] < 0.04
    np.testing.assert_array_equal(goal, TARGET)


def test_the_integral_action_brings_the_tip_to_the_target(world):
    errors, goal = run(world, regulate=True)
    assert errors[-1] < 5e-4  # from more than 2 cm
    assert np.linalg.norm(goal - TARGET) > 0.05  # the goal went where it had to
    assert np.all(np.diff(errors[-200:]) < 1e-6)  # and it is still settling, not oscillating


def test_a_step_adds_the_rate_times_the_error_to_the_goal(world):
    arm, compiled = world
    controller = vmc.VMCController(compiled)
    theta = np.linspace(0.0, 0.3, 9)
    meas = vmc.Signals(0.0, motor_position=theta, motor_velocity=np.zeros(9))
    controller.reset(0.0, meas)
    controller.step(0.0, meas)
    law = PositionRegulation(controller, {"ctrl.reach.goal": 1.0}, rate=0.25)
    tip = vmc.Kinematics(arm, coordinates="motors").position(theta, 1.0)
    target = np.array([0.1, 0.02, 0.35])
    law.step(controller, {"ctrl.reach.goal": target})
    np.testing.assert_allclose(
        controller.live_params()["ctrl.reach.goal"], TARGET + 0.25 * (target - tip), rtol=1e-12
    )


def test_a_goal_that_is_not_live_is_refused(world):
    controller = vmc.VMCController(world[1])
    with pytest.raises(ValueError, match="no live Param matches"):
        PositionRegulation(controller, {"ctrl.reach.nothing": 1.0})
