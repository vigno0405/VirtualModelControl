"""The Dynamixel plant, homing and the bus check, against a fake bus."""

import time

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.hardware import (
    DynamixelPlant,
    FakeBus,
    HardwareProfile,
    Motor,
    home,
    latency_timer,
    present_ticks,
    scan,
)
from virtualmodelcontrol.hardware.bus import (
    GOAL_CURRENT,
    GOAL_POSITION,
    OPERATING_MODE,
    RETURN_DELAY_TIME,
    STATUS_RETURN_LEVEL,
    TORQUE_ENABLE,
)
from virtualmodelcontrol.robots import adapt

PROFILE = HardwareProfile((Motor(1), Motor(2, sign=-1.0), Motor(3, mode="hold")))
START = {1: 1000, 2: 2000, 3: 3000}


def fake(ticks=START):
    return FakeBus(dict.fromkeys(ticks, "XC330-T288"), ticks)


def writes(bus, motor):
    return [(address, value) for _, (i, address, value) in bus.log if i == motor]


def test_start_sets_a_safe_goal_before_each_torque_goes_on():
    bus = fake()
    with DynamixelPlant(PROFILE, bus, watchdog=None):
        for i in START:
            assert bus.value(i, TORQUE_ENABLE) == 1
            assert bus.value(i, RETURN_DELAY_TIME) == 0 and bus.value(i, STATUS_RETURN_LEVEL) == 1
        assert [bus.value(i, OPERATING_MODE) for i in START] == [0, 0, 4]
        assert bus.value(3, GOAL_POSITION) == 3000  # the held motor stays where it started
        for i, goal in ((1, GOAL_CURRENT), (3, GOAL_POSITION)):
            log = writes(bus, i)
            on = log.index((TORQUE_ENABLE[0], 1))
            assert goal[0] in [address for address, _ in log[:on]]


def test_read_gives_angles_from_the_start_and_smoothed_rates():
    bus = fake()
    with DynamixelPlant(PROFILE, bus, watchdog=None) as plant:
        bus.set_position(1, 1000 + 1024)
        bus.set_position(2, 2000 + 512)
        bus.set_velocity(1, 100)
        first = plant.read()
        second = plant.read()
    np.testing.assert_allclose(first["motor_position"], [np.pi / 2, -np.pi / 4])
    rate = 100 * 0.229 * 2 * np.pi / 60
    np.testing.assert_allclose(first["motor_velocity"], [0.1 * rate, 0.0])
    np.testing.assert_allclose(second["motor_velocity"], [0.19 * rate, 0.0])


def test_write_sends_clamped_goal_currents_with_the_signs():
    bus = fake()
    with DynamixelPlant(PROFILE, bus, watchdog=None) as plant:
        plant.write(vmc.Signals(0.0, motor_torque=[0.1, 0.1]))
        assert [bus.value(i, GOAL_CURRENT) for i in (1, 2)] == [86, -86]
        plant.write(vmc.Signals(0.0, motor_torque=[100.0, 100.0]))
        assert [bus.value(i, GOAL_CURRENT) for i in (1, 2)] == [32767, -32767]


def test_the_watchdog_zeroes_the_torque_when_commands_stop():
    bus = fake()
    with DynamixelPlant(PROFILE, bus, watchdog=0.05) as plant:
        plant.write(vmc.Signals(0.0, motor_torque=[0.2, 0.2]))
        assert bus.value(1, GOAL_CURRENT) != 0
        time.sleep(0.3)
        assert [bus.value(i, GOAL_CURRENT) for i in (1, 2)] == [0, 0] and plant.trips == 1
        plant.write(vmc.Signals(0.0, motor_torque=[0.2, 0.2]))  # commands again: re-armed
        assert bus.value(1, GOAL_CURRENT) != 0


@pytest.mark.parametrize("error", [None, RuntimeError, KeyboardInterrupt])
def test_leaving_stops_the_motors_and_keeps_the_held_one(error):
    bus = fake()
    plant = DynamixelPlant(PROFILE, bus, watchdog=0.05)
    with pytest.raises(error) if error else _nothing(), plant:
        plant.write(vmc.Signals(0.0, motor_torque=[0.3, 0.3]))
        if error:
            raise error
    assert [bus.value(i, GOAL_CURRENT) for i in (1, 2)] == [0, 0]
    assert [bus.value(i, TORQUE_ENABLE) for i in START] == [0, 0, 1]
    assert bus.closed
    plant.close()  # a second close does nothing


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_home_moves_slowly_then_hands_over_in_torque_mode():
    bus = fake()
    assert home(bus, {1: 1500, 2: 2500}, sleep=lambda s: None)
    assert present_ticks(bus, [1, 2]) == {1: 1500, 2: 2500}
    for i in (1, 2):
        assert bus.value(i, OPERATING_MODE) == 0 and bus.value(i, GOAL_CURRENT) == 0
        assert bus.value(i, TORQUE_ENABLE) == 1
        assert (112, 40) in writes(bus, i)  # profile velocity, about 9 rpm


def test_home_refuses_a_motor_that_lost_a_turn():
    bus = fake()
    with pytest.raises(RuntimeError, match="lost turn"):
        home(bus, {1: 1000 + 4096, 2: 2000}, sleep=lambda s: None)
    assert bus.log == []  # nothing moved


def test_home_reports_a_timeout():
    class Stuck(FakeBus):
        def write(self, id, address, data):
            here = self.value(id, (132, 4))
            super().write(id, address, data)
            self.set_position(id, here)

    bus = Stuck({1: "XC330-T288"}, {1: 1000})
    clock = iter(np.arange(0.0, 100.0, 1.0))
    assert not home(bus, {1: 1200}, timeout=5.0, sleep=lambda s: None, clock=lambda: next(clock))
    assert bus.value(1, OPERATING_MODE) == 0  # still handed over in torque mode


def test_the_plant_homes_the_motors_of_its_profile():
    profile = HardwareProfile((Motor(1, home=1200), Motor(2)))
    bus = fake({1: 1000, 2: 2000})
    assert DynamixelPlant(profile, bus, watchdog=None).home(sleep=lambda s: None)
    assert present_ticks(bus, [1, 2]) == {1: 1200, 2: 2000}


def test_scan_finds_the_motors_at_their_baud_rate():
    def bus(port, baudrate):
        return fake() if baudrate == 2_000_000 else FakeBus({})

    found = scan("/dev/ttyUSB9", (57_600, 2_000_000), bus=bus)
    assert found == {2_000_000: dict.fromkeys(START, "XC330-T288")}
    assert latency_timer("/dev/no-such-port") is None


def test_one_control_step_of_the_finger_through_the_plant():
    profile = adapt.finger_hardware()
    robot, ctrl = adapt.finger(), vmc.Mechanism("ctrl")
    ctrl.add("hold", vmc.LinearSpring(robot.joint(slice(0, 2)), 1.0))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    bus = fake(dict(zip(profile.ids, adapt.FINGER_HOME, strict=True)))
    with DynamixelPlant(profile, bus, watchdog=None) as plant:
        bus.set_position(1, adapt.FINGER_HOME[0] + 200)  # the finger was pushed
        meas = plant.read()
        cmd = controller.step(meas.t, meas)
        plant.write(cmd)
    expected = profile.goal_currents(cmd["motor_torque"])
    assert [v for a, v in writes(bus, 1) if a == GOAL_CURRENT[0]][-2] == expected[0] < 0
