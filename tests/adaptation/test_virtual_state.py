"""A controller with a virtual state: the gradient is taken at the state as it is."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import ForceTracking
from virtualmodelcontrol.estimation import ContactForce
from virtualmodelcontrol.robots import helyx

NORMAL = np.array([0.2, 0.1, 0.95])


@pytest.fixture(scope="module")
def setup():
    arm = helyx.arm("145-290-290", efficiency=0.3)
    tip = arm.point(s=1.0)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("drag", vmc.LinearSpring(tip - vmc.Ref("goal", 3, value=[0.05, 0.0, 0.6]), 40.0))
    z = ctrl.add_state("follower", dim=3, unit="m", initial=[0.02, 0.0, 0.7])
    ctrl.add("mass", vmc.Inertance(z, 0.05))
    ctrl.add("tether", vmc.LinearSpring(tip - z, 25.0))
    ctrl.add("tether_damper", vmc.LinearDamper(tip - z, 0.8))
    ctrl.add("anchor", vmc.LinearSpring(z - vmc.Ref("anchor", 3, value=[0.0, 0.0, 0.5]), 10.0))
    return arm, vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl), runtime=["ctrl.*"])


def stepped(compiled):
    controller = vmc.VMCController(compiled)
    theta = np.random.default_rng(2).uniform(-0.3, 0.5, 9)
    meas = vmc.Signals(0.0, motor_position=theta, motor_velocity=np.zeros(9))
    controller.reset(0.0, meas, z0=np.concatenate([[0.03, -0.01, 0.65], np.zeros(3)]))
    controller.step(0.0, meas)
    return controller


@pytest.mark.parametrize(
    "name", ["ctrl.drag.stiffness", "ctrl.tether.stiffness", "ctrl.drag.goal", "ctrl.anchor.anchor"]
)
def test_the_gradient_matches_finite_differences_at_the_virtual_state(setup, name):
    _, compiled = setup
    controller = stepped(compiled)
    assert compiled.z0.size == 6  # a virtual state sits between the motors and the Params in x
    force = ContactForce(controller, 1.0, NORMAL)
    law = ForceTracking(controller, 1.0, name, NORMAL)
    f_meas, f_des = np.array([0.1, 0.2, 0.5]), np.array([0.0, 0.0, 0.1])
    direction = np.ravel(law.direction(controller, f_meas, f_des)[name], order="F")
    error = f_meas - f_des
    base = np.ravel(controller.live_params()[name], order="F")
    shape = compiled.params[name].shape
    numeric = np.zeros(base.size)
    for k in range(base.size):
        h = 1e-6 * max(1.0, abs(base[k]))
        values = {}
        for sign in (1, -1):
            moved = base.copy()
            moved[k] += sign * h
            controller.set({name: np.reshape(moved, shape, order="F")})
            values[sign] = force(controller)
        controller.set({name: np.reshape(base, shape, order="F")})
        numeric[k] = -error @ (values[1] - values[-1]) / (2 * h)
    if name == "ctrl.anchor.anchor":
        # it acts on the virtual state only, which the law holds fixed
        assert np.all(direction == 0.0) and np.all(np.abs(numeric) < 1e-9)
    else:
        assert np.abs(direction).max() > 1e-3
        np.testing.assert_allclose(direction, numeric, rtol=1e-6, atol=1e-9)
