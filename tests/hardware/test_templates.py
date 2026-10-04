"""Every robot template's hardware profile matches its robot and survives a YAML round trip."""

import numpy as np
import pytest

from virtualmodelcontrol.hardware import HardwareProfile
from virtualmodelcontrol.robots import adapt, bimanual, helyx, turtle

CASES = {
    "soft arm, hanging": (helyx.hardware("145-290-290"), helyx.arm("145-290-290")),
    "soft arm, side-mounted": (helyx.hardware("145-145-145"), helyx.arm("145-145-145")),
    "two arms": (bimanual.hardware(), bimanual.arms()),
    "finger": (adapt.finger_hardware(), adapt.finger()),
    "hand": (adapt.hand_hardware(), adapt.hand()),
    "turtle": (turtle.hardware(), turtle.robot()),
}


def motor_count(robot):
    if robot.actuation is None:
        return robot.space.nq
    return robot.actuation.motor_sizes(robot.space)[0]


@pytest.mark.parametrize("name", CASES)
def test_one_commanded_motor_per_motor_of_the_robot(name, tmp_path):
    profile, robot = CASES[name]
    assert len(profile.ids) == motor_count(robot)
    ids = [m.id for m in profile.motors]
    assert len(set(ids)) == len(ids)
    assert HardwareProfile.load(profile.save(tmp_path / "profile.yaml")) == profile


def test_signs_come_from_the_templates():
    assert {m.sign for m in helyx.hardware("145-290-290").motors} == {
        helyx.ENCODER_SIGN["145-290-290"]
    }
    assert tuple(m.sign for m in turtle.hardware().commanded) == turtle.MOTOR_SIGNS
    assert [m.id for m in turtle.hardware().held] == [turtle.VSA_ID]


def test_the_hand_reads_a_driver_publishing_in_id_order():
    profile = adapt.hand_hardware()
    published = np.arange(13.0)  # the value of each motor is its ID
    expected = [adapt.HAND_MOTOR_IDS[name] for name in adapt.HAND_MOTORS]
    np.testing.assert_array_equal(profile.from_bus(published), expected)
    assert [m.id for m in profile.held] == list(adapt.HAND_WRIST_IDS)
