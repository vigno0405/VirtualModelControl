"""Schedules: Params that follow values over time, swaps at given times, and a fresh start."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.control import Schedule, ScheduledController, SwapController
from virtualmodelcontrol.robots import helyx


def test_linear_and_step_values():
    ramp = Schedule("k", [(1.0, 0.0), (3.0, 10.0)])
    assert ramp.value(0.5) is None  # before the first point: the Param keeps its own value
    assert [float(ramp.value(t)) for t in (1.0, 2.0, 3.0, 9.0)] == [0.0, 5.0, 10.0, 10.0]
    steps = Schedule("goal", [(0.0, [0.0, 1.0]), (2.0, [3.0, 4.0])], "step")
    np.testing.assert_array_equal(steps.value(1.999), [0.0, 1.0])
    np.testing.assert_array_equal(steps.value(2.0), [3.0, 4.0])
    jump = Schedule("k", [(0.0, 0.0), (1.0, 1.0), (1.0, 5.0), (2.0, 5.0)])  # a jump at t = 1
    assert [float(jump.value(t)) for t in (0.5, 1.0, 1.5)] == [0.5, 5.0, 5.0]
    matrix = Schedule("K", [(0.0, np.eye(2)), (1.0, 3 * np.eye(2))])
    np.testing.assert_array_equal(matrix.value(0.5), 2 * np.eye(2))  # keeps its shape


def test_bad_schedules_are_refused():
    with pytest.raises(ValueError, match="interpolation"):
        Schedule("k", [(0.0, 1.0)], "cubic")
    with pytest.raises(ValueError, match="at least one point"):
        Schedule("k", [])
    with pytest.raises(ValueError, match="must not decrease"):
        Schedule("k", [(1.0, 0.0), (0.0, 1.0)])


def arm_controller(name="ctrl", stiffness=300.0):
    arm = helyx.add_dynamics(helyx.arm("145-145-145"))
    ctrl = vmc.Mechanism(name)
    tip = arm.point(s=1.0)
    ctrl.add("reach", vmc.LinearSpring(tip - vmc.Ref("goal", value=[0.0, 0.0, 0.435]), stiffness))
    ctrl.add("push", vmc.GaussianSpring(arm.point(s=0.5) - [0.05, 0.0, 0.2], 0.0, 0.05))
    ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    return arm, vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))


def test_a_ramp_follows_the_lab_formula_and_a_rerun_starts_afresh():
    # the lab's ramp: strength = min(1, t / T) * target, from 0
    arm, controller = arm_controller()
    T, target, dt = 0.5, 800.0, 1 / 330
    ramp = Schedule("ctrl.push.strength", [(0.0, 0.0), (T, target)])
    goal = Schedule("ctrl.reach.goal", [(0.2, [0.1, 0.0, 0.40])], "step")
    scheduled = ScheduledController(controller, [ramp, goal])
    _, reference = arm_controller()
    logs = []
    for _ in range(2):  # the same run twice
        plant, sent = vmc.sim.ModelPlant(arm), []
        scheduled.reset(plant.t, plant.read())
        reference.reset(plant.t, plant.read())
        reference.set({"ctrl.reach.goal": [0.0, 0.0, 0.435]})
        for _ in range(round(0.8 / dt)):
            t, meas = plant.t, plant.read()
            reference.set({"ctrl.push.strength": min(1.0, t / T) * target})
            if t >= 0.2:
                reference.set({"ctrl.reach.goal": [0.1, 0.0, 0.40]})
            out = scheduled.step(t, meas)["motor_torque"]
            np.testing.assert_allclose(out, reference.step(t, meas)["motor_torque"], atol=1e-12)
            sent.append(out)
            plant.write(vmc.Signals(t, motor_torque=out))
            plant.advance(dt)
        logs.append(np.array(sent))
    np.testing.assert_array_equal(logs[0], logs[1])


def test_a_swap_at_a_time_and_its_params_on_the_waiting_controller():
    arm, first = arm_controller()
    _, second = arm_controller("soft", stiffness=100.0)
    goal = Schedule("soft.reach.goal", [(0.0, [0.1, 0.0, 0.40])])  # set before it swaps in
    scheduled = ScheduledController(first, [goal], swaps=[(0.1, second, 0.2)])
    assert isinstance(scheduled.controller, SwapController)
    _, a = arm_controller()
    _, b = arm_controller("soft", stiffness=100.0)
    b.set({"soft.reach.goal": [0.1, 0.0, 0.40]})
    swap, plant, dt, swapped = SwapController(a), vmc.sim.ModelPlant(arm), 1 / 330, False
    scheduled.reset(plant.t, plant.read())
    swap.reset(plant.t, plant.read())
    for _ in range(200):
        t, meas = plant.t, plant.read()
        if not swapped and t >= 0.1:
            swap.swap(b, 0.2)
            swapped = True
        out = scheduled.step(t, meas)["motor_torque"]
        np.testing.assert_allclose(out, swap.step(t, meas)["motor_torque"], atol=1e-12)
        plant.write(vmc.Signals(t, motor_torque=out))
        plant.advance(dt)
    assert scheduled.controller.controller is second
    scheduled.reset(0.0)
    assert scheduled.controller.controller is first  # a new run starts with the first again


def test_a_param_that_is_not_live_is_refused():
    _, controller = arm_controller()
    with pytest.raises(KeyError, match="runtime"):
        ScheduledController(controller, [Schedule("ctrl.reach.s", [(0.0, 0.5)])])
