import numpy as np
import pytest

import virtualmodelcontrol as vmc
from helpers import Rod
from virtualmodelcontrol.mechanisms import LimitSpring, LinearDamper, LinearSpring, Mechanism, Ref
from virtualmodelcontrol.robots import helyx


def test_param_names_are_prefixed_and_shared_coordinates_counted_once():
    robot = Mechanism("arm", model=Rod())
    ctrl = Mechanism("ctrl")
    tip = robot.point(s=1.0)
    ctrl.add("drag", LinearSpring(tip - Ref("goal", 3), 10.0))
    ctrl.add("drag_d", LinearDamper(tip, 1.0))
    names = list(ctrl.params)
    assert names == ["drag.stiffness", "drag.s", "drag.goal", "drag_d.damping"]
    assert ctrl.params["drag.goal"].scope == "stage"
    assert list(robot.params) == ["L"]


def test_clashing_local_names_get_a_suffix():
    robot = Mechanism("arm", model=Rod())
    ctrl = Mechanism("ctrl")
    ctrl.add("pair", LinearSpring(robot.point(s=0.2) - robot.point(s=0.8), 5.0))
    assert list(ctrl.params) == ["pair.stiffness", "pair.s", "pair.s2"]


def test_mechanism_errors():
    m = Mechanism("ctrl")
    m.add("k", LinearSpring(Ref("x", 1), 1.0))
    with pytest.raises(ValueError, match="already has a component"):
        m.add("k", LinearSpring(Ref("y", 1), 1.0))
    m.add_state("phi")
    with pytest.raises(ValueError, match="already has a state"):
        m.add_state("phi")
    with pytest.raises(ValueError, match="no kinematic model"):
        m.point(s=0.5)


def test_joint_uses_the_model_unit():
    assert Mechanism("arm", model=Rod()).joint(slice(0, 2)).unit == "m"


def test_joint_counts_from_the_end_and_refuses_what_is_out_of_range():
    robot = Mechanism("arm", model=vmc.models.JointSpace(4))
    assert robot.joint(slice(0, -1)).indices == [0, 1, 2]
    assert robot.joint(slice(1, -1)).indices == [1, 2] and robot.joint(slice(None)).dim == 4
    assert robot.joint(-1).indices == [3] and robot.joint(np.int64(1)).indices == [1]
    assert robot.joint(np.array([0, 2])).indices == [0, 2]
    for bad in (slice(0, 10), slice(-5, 2), 4, -5, [0, 4]):
        with pytest.raises(ValueError, match="4 coordinates"):
            robot.joint(bad)


def test_motor_takes_a_numpy_integer():
    assert helyx.arm().motor(np.int64(2)).indices == [2]


def test_motor_angle_reads_the_actuations_angles():
    arm = helyx.arm()
    assert arm.motor(slice(None)).dim == 9 and arm.motor(3).indices == [3]
    assert arm.motor([0, 2]).indices == [0, 2] and arm.motor(slice(0, 3)).unit == "rad"
    ctrl = Mechanism("ctrl")
    ctrl.add("stop", LimitSpring(arm.motor(slice(None)), 2.0, -0.5, np.inf))  # unwinding limit only
    c = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    theta = np.zeros(9)
    theta[0], theta[4], theta[7] = -0.6, -0.4, 5.0
    signals = vmc.Signals(0.0, motor_position=theta, motor_velocity=np.zeros(9))
    c.reset(0.0, signals)
    expected = np.zeros(9)
    expected[0] = 2.0 * 0.1  # 0.1 rad past the limit: pushed back, nothing for the other two
    np.testing.assert_allclose(c.step(0.0, signals)["motor_torque"], expected, atol=1e-9)


def test_motor_angle_needs_an_actuation():
    with pytest.raises(ValueError, match="no actuation"):
        Mechanism("ctrl").motor(0)
    assert helyx.unwind_limit(helyx.arm()).lower.value.max() == pytest.approx(-helyx.UNWIND_LIMIT)
