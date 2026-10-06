"""Coordinates of a frame: its rotation, its orientation error, vectors along its axes, sums."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import virtualmodelcontrol as vmc
from virtualmodelcontrol.models import SerialChain

K = np.array([30.0, 10.0, 20.0])  # [N m/rad]


def wrist():
    """Two joints turning about z, then y, and a 0.3 m tool along x."""
    chain = SerialChain(
        ["revolute", "revolute", "prismatic"],
        axes=[[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        points=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        sites={"tool": (3, [0.3, 0.0, 0.0], [0.2, -0.1, 0.3])},
    )
    robot = vmc.Mechanism("wrist", model=chain)
    robot.add("inertia", vmc.Inertance(robot.joint(slice(0, 3)), [0.1, 0.1, 1.0]))
    return robot


def torque_and_energy(robot, ctrl, q, v=(0.0, 0.0, 0.0)):
    compiled = vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))
    p, z = compiled.live_values(), np.zeros(0)
    tau = np.array(compiled.tau(q, v, z, p, 0.0)).ravel()
    energy = float(compiled.energy(q, v, z, p, 0.0)[0])
    return tau, energy


def test_the_rotation_of_a_frame_is_its_matrix_row_by_row():
    robot = wrist()
    q = np.array([0.4, -0.7, 0.1])
    ctrl = vmc.Mechanism("ctrl")
    rotation = vmc.FrameRotation(robot.model, "tool")
    assert rotation.dim == 9
    ctrl.add("probe", vmc.LinearSpring(rotation, 1.0))
    compiled = vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))
    y = np.array(compiled.forces(q, np.zeros(3), np.zeros(0), compiled.live_values(), 0.0)[0])
    R = vmc.Kinematics(robot).rotation(q, "tool")
    np.testing.assert_allclose(y.ravel(), R.ravel(), atol=1e-14)  # row by row
    del rotation


def goal_matrix(goal):
    return Rotation.from_rotvec(goal).as_matrix()


def test_the_orientation_error_is_the_rotation_from_the_goal_in_the_goals_axes():
    robot = wrist()
    goal = np.array([0.3, -0.2, 0.5])
    error = vmc.OrientationError(robot.model, "tool", goal=goal)
    assert error.dim == 3
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("spring", vmc.LinearSpring(error, 1.0))
    compiled = vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))
    kin = vmc.Kinematics(robot)
    for q in ([0.0, 0.0, 0.0], [0.4, -0.7, 0.1], [1.2, 0.5, -0.3]):
        y = np.array(
            compiled.forces(np.array(q), np.zeros(3), np.zeros(0), compiled.live_values(), 0.0)[0]
        )
        want = Rotation.from_matrix(goal_matrix(goal).T @ kin.rotation(q, "tool")).as_rotvec()
        np.testing.assert_allclose(y.ravel(), want, atol=1e-12)
    # at the goal the error is zero
    zero = vmc.OrientationError(
        robot.model, "tool", goal=Rotation.from_matrix(kin.rotation([0, 0, 0], "tool")).as_rotvec()
    )
    assert zero.goal.shape == (3,)


def test_a_rotational_spring_on_the_orientation_error_pulls_the_frame_to_the_goal():
    robot = wrist()
    q = np.array([0.4, -0.7, 0.1])
    goal = np.array([0.3, -0.2, 0.5])
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("spring", vmc.LinearSpring(vmc.OrientationError(robot.model, "tool", goal=goal), K))
    tau, energy = torque_and_energy(robot, ctrl, q)
    # the torque is minus the gradient of the energy: the error's Jacobian is the exact one
    h = 1e-6
    grad = np.zeros(3)
    for i in range(3):
        up, down = q.copy(), q.copy()
        up[i] += h
        down[i] -= h
        grad[i] = (
            torque_and_energy(robot, ctrl, up)[1] - torque_and_energy(robot, ctrl, down)[1]
        ) / (2 * h)
    np.testing.assert_allclose(tau, -grad, rtol=1e-6, atol=1e-8)
    assert energy > 0.0 and np.abs(tau).max() > 0.1


def test_in_frame_and_from_frame_turn_a_vector_into_the_axes_of_a_frame_and_back():
    robot = wrist()
    q = np.array([0.4, -0.7, 0.1])
    vector = vmc.Ref("v", 3, value=[0.3, -0.2, 0.7])
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("a", vmc.LinearSpring(vmc.InFrame(vector, robot.model, "tool"), 1.0))
    ctrl.add("b", vmc.LinearSpring(vmc.FromFrame(vector, robot.model, "tool"), 1.0))
    ctrl.add(
        "c",
        vmc.LinearSpring(
            vmc.FromFrame(vmc.InFrame(vector, robot.model, "tool"), robot.model, "tool"), 1.0
        ),
    )
    compiled = vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))
    a, b, c = (
        np.array(y).ravel()
        for y in compiled.forces(q, np.zeros(3), np.zeros(0), compiled.live_values(), 0.0)[::4]
    )
    R = vmc.Kinematics(robot).rotation(q, "tool")
    np.testing.assert_allclose(a, R.T @ [0.3, -0.2, 0.7], atol=1e-14)
    np.testing.assert_allclose(b, R @ [0.3, -0.2, 0.7], atol=1e-14)
    np.testing.assert_allclose(c, [0.3, -0.2, 0.7], atol=1e-14)


def test_a_stiffness_given_along_the_axes_of_a_frame_follows_the_frame():
    robot = wrist()
    q = np.array([0.4, -0.7, 0.1])
    ctrl = vmc.Mechanism("ctrl")
    tip = robot.point("tool")
    offset = vmc.Ref("offset", 3, value=[0.01, 0.02, -0.01])
    ctrl.add(
        "spring",
        vmc.LinearSpring(vmc.InFrame(tip - offset, robot.model, "tool"), [200.0, 50.0, 10.0]),
    )
    tau, energy = torque_and_energy(robot, ctrl, q)
    R = vmc.Kinematics(robot).rotation(q, "tool")
    delta = vmc.Kinematics(robot).position(q, "tool") - np.array([0.01, 0.02, -0.01])
    local = R.T @ delta
    assert energy == pytest.approx(
        0.5 * (200 * local[0] ** 2 + 50 * local[1] ** 2 + 10 * local[2] ** 2), rel=1e-12
    )
    assert abs(tau).max() > 1e-3


def test_a_sum_of_coordinates_adds_them_and_the_plus_sign_makes_one():
    robot = wrist()
    q = np.array([0.4, -0.7, 0.1])
    tip = robot.point("tool")
    ctrl = vmc.Mechanism("ctrl")
    both = tip + np.array([0.0, 0.0, 0.05])
    assert isinstance(both, vmc.Sum) and both.dim == 3
    ctrl.add("spring", vmc.LinearSpring(both, 10.0))
    tau, energy = torque_and_energy(robot, ctrl, q)
    pos = vmc.Kinematics(robot).position(q, "tool") + np.array([0.0, 0.0, 0.05])
    assert energy == pytest.approx(5.0 * pos @ pos, rel=1e-12)
    with pytest.raises(ValueError, match="cannot add"):
        vmc.Sum(tip, vmc.Ref("x", 2))
    assert abs(tau).max() > 1e-3


def test_the_units_and_the_checks_of_the_frame_coordinates():
    robot = wrist()
    tip = robot.point("tool")
    assert vmc.OrientationError(robot.model, "tool").unit == "rad"
    assert vmc.FrameRotation(robot.model, "tool").unit == ""
    assert vmc.InFrame(tip, robot.model, "tool").unit == "m"  # the vector's own
    assert vmc.OrientationError(robot.model, "tool").goal.unit == "rad"
    for make in (vmc.FrameRotation, vmc.OrientationError):
        with pytest.raises(ValueError, match="site name"):
            make(robot.model)
    with pytest.raises(ValueError, match="site name"):
        vmc.InFrame(tip, robot.model)
    with pytest.raises(ValueError, match="3 entries, the coordinate has 2"):
        vmc.InFrame(vmc.Ref("x", 2), robot.model, "tool")


def test_a_point_with_an_offset_is_turned_with_its_frame():
    robot = wrist()
    q = np.array([0.4, -0.7, 0.1])
    kin = vmc.Kinematics(robot)
    offset = np.array([0.0, 0.05, 0.02])
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("probe", vmc.LinearSpring(robot.point("tool", offset=offset), 1.0))
    compiled = vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl))
    y = compiled.forces(q, np.zeros(3), np.zeros(0), compiled.live_values(), 0.0)[0]
    want = kin.position(q, "tool") + kin.rotation(q, "tool") @ offset
    np.testing.assert_allclose(np.array(y).ravel(), want, atol=1e-14)
    assert "offset" in robot.point("tool", offset=offset).params()
