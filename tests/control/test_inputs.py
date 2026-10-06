import numpy as np

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
