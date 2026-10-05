"""Replays: a run's commands sent again to a simulated robot, and the comparison with the run."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import adapt, helyx
from virtualmodelcontrol.sim import ModelPlant, RunLog, compare, replay

SIGNALS = ("q", "v", "motor_position", "motor_velocity")


def soft_arm(stiffness=None):
    return helyx.add_dynamics(helyx.arm("145-145-145"), stiffness=stiffness)


def run_soft_arm(arm, T=0.5):
    tip = arm.point(s=1.0)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(tip - [0.05, 0.0, 0.40], 300.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    return vmc.sim.run(ModelPlant(arm), controller, vmc.sim.SimClock(1 / 330), T=T)


def finger():
    return adapt.add_dynamics(adapt.finger(), damping=0.01)


def run_finger(T=0.3):
    robot = finger()
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("hold", vmc.LinearSpring(robot.joint(slice(0, 2)) - [0.3, 0.3], 1.0))
    ctrl.add("damp", vmc.LinearDamper(robot.joint(slice(0, 2)), 0.05))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    plant = ModelPlant(robot, q0=[0.8, 0.8], max_step=1e-4)
    return vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 500), T=T)


def motors_only(log):
    """What a real robot's log holds: the time, the commands and the motors' readings."""
    keep = ("t", "motor_torque", "motor_position", "motor_velocity")
    rows = log.arrays()
    out = RunLog()
    for k in range(len(rows["t"])):
        out.step(**{name: rows[name][k] for name in keep})
    return out


def test_a_replay_on_the_true_model_gives_the_run_back(tmp_path):
    log = RunLog.load(run_soft_arm(soft_arm()).save(tmp_path / "run"))  # as a saved file
    assert np.ptp(log.arrays()["q"], axis=0).max() > 1e-3  # the arm moves
    again = replay(log, ModelPlant(soft_arm()))
    np.testing.assert_array_equal(again.arrays()["t"], log.arrays()["t"])
    np.testing.assert_array_equal(again.arrays()["motor_torque"], log.arrays()["motor_torque"])
    gap = compare(log, again, names=SIGNALS)
    for name in SIGNALS:
        assert gap[name]["max"] < 1e-9, name


CASES = {  # a run, the robot it ran on, and the simulator's step
    "arm": (lambda: run_soft_arm(soft_arm()), soft_arm, 1e-3),
    "finger": (run_finger, finger, 1e-4),
}


@pytest.mark.parametrize("case", CASES)
def test_a_log_of_the_motors_alone_replays_from_their_first_reading(case):
    make_run, make_robot, max_step = CASES[case]
    log = make_run()
    real = motors_only(log)
    assert sorted(real.arrays()) == ["motor_position", "motor_torque", "motor_velocity", "t"]
    again = replay(real, ModelPlant(make_robot(), max_step=max_step))
    gap = compare(log, again, names=SIGNALS)
    for name in SIGNALS:
        assert gap[name]["max"] < 1e-9, name


def test_a_wrong_model_shows_in_the_comparison():
    log = run_soft_arm(soft_arm())
    stiff = replay(log, ModelPlant(soft_arm(stiffness=1.5 * helyx.SIM_STIFFNESS)))
    gap = compare(log, stiff, names=["q", "motor_position"])
    assert gap["q"]["max"] > 1e-4 and 0.0 < gap["q"]["rms"] < gap["q"]["max"]
    assert gap["motor_position"]["max"] > 1e-4
    assert (
        gap["q"]["max"] > 1e5 * compare(log, replay(log, ModelPlant(soft_arm())), ["q"])["q"]["max"]
    )


def test_each_command_is_held_for_as_long_as_it_was_in_the_run():
    arm, rng = soft_arm(), np.random.default_rng(3)
    times = np.array([0.0, 0.004, 0.011, 0.02, 0.026])  # unevenly spaced, as a real loop's are
    torques = rng.uniform(-0.05, 0.05, (len(times), 9))
    plant, real = ModelPlant(arm), RunLog()
    for k, (time, torque) in enumerate(zip(times, torques, strict=True)):
        meas = plant.read()
        plant.write(vmc.Signals(plant.t, motor_torque=torque))
        real.step(t=time, motor_torque=torque, **{n: meas[n] for n in meas.names})
        if k + 1 < len(times):
            plant.advance(times[k + 1] - time)
    replica = ModelPlant(arm)
    again = replay(real, replica)
    assert compare(real, again, names=SIGNALS)["q"]["max"] < 1e-12
    assert replica.t == pytest.approx(times[-1] - times[0])  # the last command has no duration
    assert np.ptp(real.arrays()["q"], axis=0).max() > 1e-6  # the commands moved the arm
