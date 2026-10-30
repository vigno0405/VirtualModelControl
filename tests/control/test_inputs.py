import numpy as np
import pytest

import virtualmodelcontrol as vmc


def controller():
    robot = vmc.Mechanism("robot", model=vmc.models.JointSpace(2))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("hold", vmc.LinearSpring(robot.joint(slice(0, 2)) - vmc.Ref("goal", 2), 5.0))
    return vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))


def meas(t):
    return vmc.Signals(t, motor_position=[0.1, -0.2], motor_velocity=[0.5, 0.0])


def test_there_are_no_inputs_before_the_first_step():
    assert controller().inputs() is None


def test_the_inputs_are_what_the_last_step_read_with_the_params_as_they_are():
    c = controller()
    c.step(0.0, meas(0.0))
    c.step(0.5, meas(0.5))
    x = c.inputs()
    np.testing.assert_array_equal(x[:4], [0.1, -0.2, 0.5, 0.0])  # angles and rates
    assert x[-1] == 0.5  # time since the start
    c.set({"ctrl.hold.goal": [1.0, 2.0]})  # after the step
    y = c.inputs()
    np.testing.assert_array_equal(y[:4], x[:4])
    np.testing.assert_array_equal(y[-3:-1], [1.0, 2.0])  # the new goal is in the vector
    assert not np.array_equal(x, y)


def test_set_checks_the_size_of_every_value():
    c = controller()
    c.step(0.0, meas(0.0))
    with pytest.raises(ValueError, match=r"'ctrl.hold.goal' has shape \(2,\), got .* shape \(\)"):
        c.set({"ctrl.hold.goal": 1.0})  # a scalar does not fill a vector
    with pytest.raises(ValueError, match=r"'ctrl.hold.goal' has shape \(2,\), got .* shape \(3,\)"):
        c.set({"ctrl.hold.goal": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match=r"'ctrl.hold.stiffness' has shape \(\), got .* \(2,\)"):
        c.set({"ctrl.hold.stiffness": [1.0, 2.0]})
    np.testing.assert_array_equal(c.live_params()["ctrl.hold.goal"], [0.0, 0.0])  # nothing moved
    c.set({"ctrl.hold.stiffness": 7.0, "ctrl.hold.goal": [[1.0], [2.0]]})  # the right size
    assert c.live_params()["ctrl.hold.stiffness"] == 7.0
    np.testing.assert_array_equal(c.live_params()["ctrl.hold.goal"], [1.0, 2.0])
    robot = vmc.Mechanism("robot", model=vmc.models.JointSpace(2))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("hold", vmc.LinearSpring(robot.joint(slice(0, 2)), np.diag([1.0, 2.0])))
    matrix = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    with pytest.raises(ValueError, match=r"'ctrl.hold.stiffness' has shape \(2, 2\)"):
        matrix.set({"ctrl.hold.stiffness": 100.0})
    matrix.set({"ctrl.hold.stiffness": [1.0, 0.0, 0.0, 3.0]})  # as many entries as the matrix
    np.testing.assert_array_equal(matrix.live_params()["ctrl.hold.stiffness"], np.diag([1.0, 3.0]))


def test_live_params_are_copies():
    c = controller()
    c.step(0.0, meas(0.0))
    energy = c.energy()
    live = c.live_params()
    live["ctrl.hold.stiffness"] *= 10.0  # editing what was returned is not a set
    live["ctrl.hold.goal"][:] = 9.0
    assert c.live_params()["ctrl.hold.stiffness"] == 5.0 and c.energy() == energy
    np.testing.assert_array_equal(c.live_params()["ctrl.hold.goal"], [0.0, 0.0])


def test_the_balance_and_the_elements_use_the_params_as_they_are_after_a_set():
    c = controller()
    c.step(0.0, meas(0.0))
    before = c.elements()["ctrl.hold"]["force"]
    c.set({"ctrl.hold.stiffness": 50.0})
    assert c.balance()["stored"] == pytest.approx(c.energy())
    np.testing.assert_allclose(c.elements()["ctrl.hold"]["force"], 10.0 * before)
