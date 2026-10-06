import numpy as np
import pytest

import virtualmodelcontrol as vmc
from underactuated_lab import ROBOTS, build, lab, to_3d
from virtualmodelcontrol.robots import planar

DATA = lab()


@pytest.mark.parametrize("prefix", ROBOTS)
def test_models_match_the_lab(prefix):
    robot = build(prefix)
    dynamics, kin = vmc.compile_dynamics(robot), vmc.Kinematics(robot)
    p = dynamics.live_values()
    B = DATA[f"{prefix}_B"]
    np.testing.assert_array_equal(robot.actuation.params["B"].value, B)
    q, dq, tau = (DATA[f"{prefix}_{k}"] for k in ("q", "dq", "tau"))
    start = None
    for i in range(len(q)):
        mass = np.array(dynamics.mass(q[i], p))
        np.testing.assert_allclose(mass, DATA[f"{prefix}_M"][i], rtol=1e-8, atol=1e-9)
        u = np.linalg.pinv(B) @ tau[i]
        a = np.array(dynamics.forward(q[i], dq[i], u, p, 0.0)).ravel()
        # the soft arm's lab Coriolis term is a finite difference
        np.testing.assert_allclose(a, DATA[f"{prefix}_ddq"][i], rtol=1e-4, atol=2e-5)
        T, V = (float(x) for x in dynamics.energy(q[i], dq[i], p, 0.0))
        assert T == pytest.approx(DATA[f"{prefix}_ke"][i], rel=1e-8, abs=1e-9)
        start = V - DATA[f"{prefix}_pe"][i] if start is None else start  # the lab's zero differs
        assert V - DATA[f"{prefix}_pe"][i] == pytest.approx(start, abs=1e-8)
        np.testing.assert_allclose(
            kin.position(q[i], "tip"), to_3d(prefix, DATA[f"{prefix}_tip"][i]), atol=1e-8
        )
        J = kin.jacobian(q[i], "tip")
        lab_J = DATA[f"{prefix}_J"][i]
        expected = np.insert(lab_J, 1 if prefix == "helyx" else 2, 0.0, axis=0)
        np.testing.assert_allclose(J, expected, atol=1e-8)


@pytest.mark.parametrize("prefix", ROBOTS)
def test_the_templates_keep_the_model_contract(prefix):
    robot = build(prefix)
    vmc.testing.check_model(robot, at=["tip"], samples=2)


def test_the_passive_joints_are_those_without_a_motor():
    assert planar.passive_joints(planar.arm("three-link")) == [1]
    assert planar.passive_joints(planar.arm("five-link")) == [1, 2, 3]
    assert planar.passive_joints(planar.arm("five-link", actuated=(0, 1, 4))) == [2, 3]
    B = planar.continuum().actuation.params["B"].value
    np.testing.assert_allclose(B, [[-2, -2], [-2, 0], [-2, 0]])
    assert np.linalg.matrix_rank(B) == 2  # the 2nd against the 3rd section has no motor


def test_arguments_reach_the_arm():
    arm = planar.arm(lengths=(0.2, 0.3), masses=(1.0, 2.0), actuated=(1,), rest=[0.1, 0.2],
                     gravity=(0.0, 0.0, -9.81), efficiency=0.5)  # fmt: skip
    assert arm.model.space.nq == 2 and arm.actuation.motor_sizes(arm.model.space) == (1, 1)
    np.testing.assert_allclose(vmc.Kinematics(arm).position([0.0, 0.0], "tip"), [0.5, 0, 0])
    assert float(arm.params["m2.mass"].value) == 2.0
    assert float(arm.params["I2.inertia"].value[1, 1]) == pytest.approx(2.0 * 0.09 / 12)
    np.testing.assert_allclose(arm.params["q_rest"].value, [0.1, 0.2])
    assert float(arm.params["efficiency.c1"].value) == 0.5
    arm = planar.add_dynamics(arm, stiffness=3.0, damping=0.1)
    np.testing.assert_allclose(arm.params["spring.rest"].value, [0.1])  # joint 1 only
    with pytest.raises(ValueError, match="masses"):
        planar.arm(lengths=(0.2, 0.3), masses=(1.0,))
    with pytest.raises(KeyError):
        planar.arm("four-link")


def test_the_defaults_are_the_named_presets():
    for name, spec in planar.PRESETS.items():
        named, explicit = planar.arm(name), planar.arm(name, lengths=spec["lengths"])
        np.testing.assert_array_equal(named.params.vector(), explicit.params.vector())


def test_the_continuum_arm_is_an_arc_of_fixed_length():
    arm = planar.continuum()
    kin = vmc.Kinematics(arm)
    np.testing.assert_allclose(kin.position(np.zeros(3), "tip"), [0, 0, 0.75], atol=1e-9)
    phi = 0.8  # one section of 0.25 m bent by 0.8 rad
    tip = kin.position([phi, 0.0, 0.0], "seg1")
    np.testing.assert_allclose(tip, [-0.25 * (1 - np.cos(phi)) / phi, 0, 0.25 * np.sin(phi) / phi],
                               atol=1e-9)  # fmt: skip
    custom = planar.continuum(lengths=(0.1, 0.2), masses=(0.1, 0.1), offsets=(0.01,), depths=(2,))
    assert custom.model.space.nq == 2
    np.testing.assert_allclose(custom.actuation.params["B"].value, [[-1.0], [-1.0]])
