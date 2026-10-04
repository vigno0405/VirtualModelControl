"""Smooth element swaps: the quintic blend of the lab's controller, and swaps in a run."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.control import SwapController, blend_weight
from virtualmodelcontrol.robots import adapt

# The lab's blend weight at these times, for a 1 s blend.
LAB = {-0.5: 0.0, 0.0: 0.0, 0.1: 0.00856, 0.25: 0.103515625, 0.5: 0.5, 0.75: 0.896484375, 1.0: 1.0}


def test_blend_weight_matches_the_lab_and_is_flat_at_both_ends():
    for t, w in LAB.items():
        assert blend_weight(t, 1.0) == pytest.approx(w, abs=1e-15)
    assert blend_weight(1.5, 1.0) == 1.0 and blend_weight(0.3, 0.0) == 1.0
    assert blend_weight(0.5, 2.0) == pytest.approx(0.103515625)
    s = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose([blend_weight(x, 1.0) + blend_weight(1 - x, 1.0) for x in s], 1.0)
    h = 1e-4
    for end in (0.0, 1.0):  # first and second derivatives vanish at both ends
        d1 = (blend_weight(end + h, 1.0) - blend_weight(end - h, 1.0)) / (2 * h)
        d2 = blend_weight(end + h, 1.0) - 2 * blend_weight(end, 1.0) + blend_weight(end - h, 1.0)
        assert abs(d1) < 1e-6 and abs(d2 / h**2) < 1e-2


def finger_controller(robot, goal):
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("tip", vmc.LinearSpring(robot.point("tip") - goal, 100.0))
    ctrl.add("damp", vmc.LinearDamper(robot.point("tip"), 1.0))
    ctrl.add("gravity", vmc.GravityCompensation(robot))
    return vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))


def test_a_swap_blends_the_two_controllers_then_hands_over():
    robot = adapt.add_dynamics(adapt.finger(), damping=0.01)
    goals = ([0.0, 0.04, 0.06], [0.0, 0.06, 0.03], [0.0, 0.05, 0.05])
    a, b, c = (finger_controller(robot, g) for g in goals)
    ref = [finger_controller(robot, g) for g in goals]  # fresh copies give the expected torques
    swap, plant, dt = SwapController(a), vmc.sim.ModelPlant(robot), 1 / 500
    swap.reset(0.0, plant.read())
    sent = []
    for k in range(400):
        t, meas = plant.t, plant.read()
        if k == 50:
            swap.swap(b, 0.2)
        if k == 100:
            swap.swap(c, 0.1)  # asked during the first blend: waits for it
        out = swap.step(t, meas)
        u = [r.step(t, meas)["motor_torque"] for r in ref]
        if k < 50:
            expected = u[0]
        elif k <= 150:
            w = blend_weight(t - 50 * dt, 0.2)
            expected = (1 - w) * u[0] + w * u[1]
        elif k <= 200:
            w = blend_weight(t - 151 * dt, 0.1)
            expected = (1 - w) * u[1] + w * u[2]
        else:
            expected = u[2]
        np.testing.assert_allclose(out["motor_torque"], expected, atol=1e-12, err_msg=f"step {k}")
        sent.append(out["motor_torque"])
        plant.write(out)
        plant.advance(dt)
    assert swap.controller is c and not swap.blending
    jumps = np.abs(np.diff(np.array(sent), axis=0)).max(axis=1)
    assert jumps[49:52].max() < 5 * np.median(jumps[:49]) + 1e-9  # no jump when the swap starts
