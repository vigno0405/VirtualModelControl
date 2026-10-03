import pytest

from helpers import Rod
from virtualmodelcontrol.mechanisms import LinearDamper, LinearSpring, Mechanism, Ref


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
