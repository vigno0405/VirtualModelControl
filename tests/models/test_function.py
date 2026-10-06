"""A model written as a function: it works everywhere a model does, and passes the checks."""

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.testing import check_model


def pendulum_frame(q, at, p):
    c, s = ca.cos(q[0]), ca.sin(q[0])
    R = ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))
    return R, p["L"] * ca.vertcat(c, s, 0)


def pendulum(length=0.5):
    params = [vmc.Param("L", length, unit="m", scope="design")]
    return vmc.models.FunctionModel(pendulum_frame, 1, params, sites=("bob",))


def test_a_function_model_gives_positions_rotations_and_derivatives():
    kin, q = vmc.Kinematics(pendulum(0.5)), [0.7]
    c, s = np.cos(0.7), np.sin(0.7)
    np.testing.assert_allclose(kin.position(q, "bob"), 0.5 * np.array([c, s, 0]))
    np.testing.assert_allclose(kin.jacobian(q, "bob").ravel(), 0.5 * np.array([-s, c, 0]))
    np.testing.assert_allclose(kin.hessian(q, "bob").ravel(), -0.5 * np.array([c, s, 0]))
    assert kin.rotation(q, "bob")[0, 1] == pytest.approx(-s)


def test_the_params_of_the_function_are_the_models_and_can_change():
    model = pendulum(0.5)
    assert list(model.params) == ["L"] and model.sites == ("bob",) and model.q_unit == "rad"
    kin = vmc.Kinematics(model)
    before = kin.position([0.0], "bob")[0]
    model.params["L"].value = 0.8
    assert kin.position([0.0], "bob")[0] == pytest.approx(0.8) and before == pytest.approx(0.5)


def test_a_function_model_passes_the_checks_and_has_nothing_to_serialize():
    robot = vmc.Mechanism("pendulum", model=pendulum())
    robot.add_param(vmc.Param("gravity", [0.0, -9.81, 0.0], unit="m/s^2"))
    robot.add("bob", vmc.PointMass(robot.point("bob"), 0.2))
    robot.add("weight", vmc.Gravity(robot))
    worst = check_model(robot)
    assert "serialization" not in worst and worst["energy"] == 0.0
    assert not hasattr(pendulum(), "to_dict")


def test_a_function_model_drives_a_controller_and_a_simulation():
    robot = vmc.Mechanism("pendulum", model=pendulum())
    robot.add_param(vmc.Param("gravity", [0.0, -9.81, 0.0], unit="m/s^2"))
    robot.add("bob", vmc.PointMass(robot.point("bob"), 0.2))
    robot.add("weight", vmc.Gravity(robot))
    robot.add("friction", vmc.LinearDamper(robot.joint(0), 0.05))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("hold", vmc.LinearSpring(robot.point("bob") - [0.0, 0.5, 0.0], 50.0))
    ctrl.add("damp", vmc.LinearDamper(robot.point("bob"), 1.0))
    ctrl.add("gravity", vmc.GravityCompensation(robot))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    plant = vmc.sim.ModelPlant(robot, q0=[0.2])
    vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 500), T=4.0)
    assert plant.q[0] == pytest.approx(np.pi / 2, abs=0.02)  # the bob is held above the pivot


def test_arrays_and_lists_are_accepted_as_frames_and_a_space_by_number_or_object():
    def slider(q, at, p):  # a prismatic joint along x: a numpy rotation, a list for the position
        return np.eye(3), [q[0], 0.0, 0.0]

    model = vmc.models.FunctionModel(slider, vmc.Euclidean(1), q_unit="m")
    kin = vmc.Kinematics(model)
    np.testing.assert_allclose(kin.position([0.3], 0), [0.3, 0.0, 0.0])
    np.testing.assert_allclose(kin.jacobian([0.3], 0), [[1.0], [0.0], [0.0]])
    assert model.space.nq == 1 and model.sites == () and len(model.params) == 0


def test_a_rotation_can_be_a_list_of_rows():
    def spinner(q, at, p):
        c, s = ca.cos(q[0]), ca.sin(q[0])
        return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], (0.0, 0.0, 0.1)

    kin = vmc.Kinematics(vmc.models.FunctionModel(spinner, 1, q_unit="rad"))
    np.testing.assert_allclose(
        kin.rotation([0.3], 0)[:2, :2], [[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]
    )
    np.testing.assert_allclose(kin.position([0.3], 0), [0.0, 0.0, 0.1])
