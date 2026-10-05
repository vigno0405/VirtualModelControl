"""A gate scales an element's force and energy; it is a live Param between 0 and 1."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from helpers import component_function, evaluate
from virtualmodelcontrol.mechanisms import Gated, Joint, LinearDamper, LinearSpring, TanhSpring
from virtualmodelcontrol.models import JointSpace

Y3 = Joint(slice(0, 3), unit="m")
rng = np.random.default_rng(5)
ELEMENTS = {
    "linear": lambda: LinearSpring(Y3, [10.0, 20.0, 30.0]),
    "tanh": lambda: TanhSpring(Y3, 65.0, 0.5),
    "damper": lambda: LinearDamper(Y3, 1.5),
}


@pytest.mark.parametrize("name", ELEMENTS)
def test_the_gate_scales_the_force_and_the_energy(name):
    plain, gated = ELEMENTS[name](), Gated(ELEMENTS[name](), 0.3)
    assert gated.kind == plain.kind
    q, v = rng.uniform(-0.2, 0.2, (20, 3)), rng.uniform(-1.0, 1.0, (20, 3))
    for output in range(5):  # Jᵀf, f, y, ẏ, V
        scale = 0.3 if output in (0, 1, 4) else 1.0
        expected = scale * evaluate(component_function(plain, 3), q, v, output)
        np.testing.assert_allclose(
            evaluate(component_function(gated, 3), q, v, output), expected, atol=1e-14
        )


def test_the_force_of_a_gated_spring_is_minus_the_gradient_of_its_energy():
    fn = component_function(Gated(TanhSpring(Y3, 65.0, 0.5), 0.4), 3)
    h = 1e-6
    for y in rng.uniform(-0.2, 0.2, (10, 3)):
        f = np.array(fn(y, np.zeros(3))[1]).ravel()
        V = lambda x: float(fn(x, np.zeros(3))[4])  # noqa: E731
        grad = np.array([(V(y + h * e) - V(y - h * e)) / (2 * h) for e in np.eye(3)])
        np.testing.assert_allclose(f, -grad, rtol=1e-6, atol=1e-8)


def test_the_gate_is_a_live_param_between_zero_and_one():
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    robot.add("mass", vmc.Inertance(robot.joint(0), 1.0))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("spring", Gated(LinearSpring(robot.joint(0) - 1.0, 5.0), 0.5))
    system = vmc.VirtualMechanismSystem(robot, ctrl)
    gate = system.params["ctrl.spring.gate"]
    assert gate.bounds == (0.0, 1.0) and gate.scope == "stage"
    assert "ctrl.spring.stiffness" in system.params  # the element's own Params stay reachable
    controller = vmc.VMCController(vmc.compile(system))
    meas = vmc.Signals(0.0, motor_position=[0.0], motor_velocity=[0.0])
    controller.reset(0.0, meas)
    assert controller.step(0.0, meas)["motor_torque"] == pytest.approx([0.5 * 5.0])
    controller.set({"ctrl.spring.gate": 0.0})
    assert controller.step(0.0, meas)["motor_torque"] == pytest.approx([0.0])
    controller.set({"ctrl.spring.gate": 1.0})
    assert controller.step(0.0, meas)["motor_torque"] == pytest.approx([5.0])


def test_an_inertance_has_no_force_to_gate():
    with pytest.raises(TypeError, match="inertance"):
        Gated(vmc.Inertance(Y3, 1.0))


def test_the_gate_is_in_the_representation():
    assert "gate=0.5" in repr(Gated(LinearSpring(Y3, 10.0), 0.5))
