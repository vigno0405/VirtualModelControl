"""The compiled controller against torques recorded from the original lab controller.

145/290/290 arm; the recorded motor angles have the opposite sign to the library's (θ > 0 pulls).
"""

from pathlib import Path

import casadi as ca
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.core import constants
from virtualmodelcontrol.robots import helyx

DATA = np.load(Path(__file__).parents[1] / "data" / "vmc.npz")
SIGN = helyx.ENCODER_SIGN["145-290-290"]


def three_elements():
    arm = helyx.arm("145-290-290")
    ctrl = vmc.Mechanism("ctrl")
    p0, p1, p2 = arm.point(s=0.2), arm.point(s=0.6), arm.point(s=1.0)
    ctrl.add("k0", vmc.LinearSpring(p0 - vmc.Ref("goal", 3), 30.0))
    ctrl.add("c0", vmc.LinearDamper(p0, 1.5))
    ctrl.add("k1", vmc.TanhSpring(p1 - vmc.Ref("goal", 3), 65.0, 0.5))
    ctrl.add("c1", vmc.LinearDamper(p1, 1.0))
    ctrl.add("k2", vmc.GaussianSpring(p2 - vmc.Ref("goal", 3), 50.0, 0.06))
    ctrl.add("c2", vmc.TanhDamper(p2, 0.33, 0.5))
    return arm, ctrl


def measure(i):
    return vmc.Signals(0.0, motor_position=SIGN * DATA["q"][i], motor_velocity=SIGN * DATA["qd"][i])


def test_torques_match_the_recorded_controller():
    arm, ctrl = three_elements()
    law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
    controller = vmc.VMCController(law)
    for i in range(len(DATA["q"])):
        goals = DATA["targets"][i]
        controller.set({f"ctrl.k{k}.goal": goals[k] for k in range(3)})
        u = controller.step(0.0, measure(i))["motor_torque"]
        np.testing.assert_allclose(SIGN * u, DATA["tau"][i], rtol=1e-10, atol=1e-13)


def test_per_component_forces_positions_and_velocities():
    arm, ctrl = three_elements()
    law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
    names = law.component_names
    p_act = constants(arm.actuation.params)
    slices = law.live_slices()
    for i in range(len(DATA["q"])):
        q = arm.actuation.config_from_motors(ca.DM(SIGN * DATA["q"][i]), p_act)
        v = arm.actuation.config_from_motors(ca.DM(SIGN * DATA["qd"][i]), p_act)
        p = law.live_values()
        for k in range(3):
            p[slices[f"ctrl.k{k}.goal"]] = DATA["targets"][i][k]
        out = [np.array(x).ravel() for x in law.forces(q, v, [], p, 0.0)]

        def get(name, j, out=out):  # outputs per component: y, ẏ, f, τ
            return out[4 * names.index(name) + j]

        for k in range(3):
            np.testing.assert_allclose(get(f"ctrl.c{k}", 0), DATA["pos"][i][k], atol=1e-14)
            np.testing.assert_allclose(get(f"ctrl.c{k}", 1), DATA["vel"][i][k], atol=1e-13)
            force = get(f"ctrl.k{k}", 2) + get(f"ctrl.c{k}", 2)
            np.testing.assert_allclose(force, DATA["F_comp"][i][k], rtol=1e-10, atol=1e-13)


def test_gravity_compensation_matches_the_recorded_torques():
    arm = helyx.arm("145-290-290")
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    for i in range(len(DATA["q"])):
        u = controller.step(0.0, measure(i))["motor_torque"]
        np.testing.assert_allclose(SIGN * u, DATA["tau_grav"][i], rtol=1e-10, atol=1e-14)
