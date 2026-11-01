"""The underactuated controllers against the lab's own code, on its three robots."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from underactuated_lab import (
    COMBOS,
    ROBOTS,
    build,
    lab,
    lab_controller,
    law,
    measurement,
    to_3d,
)
from virtualmodelcontrol.control import StateController
from virtualmodelcontrol.control import underactuated as ua

DATA = lab()


def reading(prefix, robot, i):
    q, v = DATA[f"{prefix}_q"][i], DATA[f"{prefix}_dq"][i]
    return measurement(robot, q, v), q, v


@pytest.mark.parametrize("gravity", [False, True])
@pytest.mark.parametrize("prefix", ROBOTS)
def test_frozen_point(prefix, gravity):
    robot = build(prefix)
    frozen = (
        ua.Moving(robot.actuation, vmc.compile_dynamics(robot))
        if gravity
        else ua.Frozen(robot.actuation)
    )
    for i, expected in enumerate(DATA[f"{prefix}_g{int(gravity)}_frozen_point"]):
        meas, _, v = reading(prefix, robot, i)
        point, rate = frozen(meas)
        np.testing.assert_allclose(point, expected, atol=1e-7)
        if not gravity:  # a moving rest has the velocity P theta_dot instead
            B = frozen.B
            np.testing.assert_allclose(rate, B @ np.linalg.pinv(B) @ v, atol=1e-12)


@pytest.mark.parametrize("gravity", [False, True])
@pytest.mark.parametrize("base,correction", COMBOS)
@pytest.mark.parametrize("prefix", ROBOTS)
def test_every_flag_against_the_lab(prefix, base, correction, gravity):
    robot = build(prefix)
    compiled = law(prefix, robot, gravity)
    controller = lab_controller(compiled, robot, base, correction, gravity)
    tag = f"{prefix}_g{int(gravity)}_{base}_{correction or 'none'}"
    for i in range(len(DATA[f"{prefix}_q"])):
        meas = reading(prefix, robot, i)[0]
        controller.reset(0.0)
        if correction is not None:
            controller.output[0].level = DATA[f"{prefix}_T"][i] if correction == "tank" else None
        command = controller.step(0.0, meas)
        np.testing.assert_allclose(command["law_torque"], DATA[f"{tag}_u_base"][i], atol=1e-7)
        np.testing.assert_allclose(command["motor_torque"], DATA[f"{tag}_u"][i], atol=1e-7)
        if correction is not None:
            stage = controller.output[0]
            np.testing.assert_allclose(stage.dissipation, DATA[f"{tag}_Vdot_D"][i], atol=1e-10)
            np.testing.assert_allclose(stage.alpha, DATA[f"{tag}_alpha"][i], atol=1e-7)
            if correction == "tank":
                np.testing.assert_allclose(stage.gate, DATA[f"{tag}_gate"][i], atol=1e-9)


def test_the_flags_pick_the_controller():
    robot = build("three")
    compiled = law("three", robot)
    assert type(ua.controller(compiled, "frozen")) is vmc.VMCController
    assert isinstance(ua.controller(compiled, "naive"), StateController)
    assert isinstance(ua.controller(compiled, "frozen", rest="moving"), StateController)
    assert isinstance(ua.controller(compiled, "frozen", at=[0.1, 0.2]), StateController)
    assert ua.controller(compiled, "naive").output == []
    stage = ua.controller(compiled, "naive", "tank", tank=2.5, width=0.5).output[0]
    assert (stage.level, stage.width) == (2.5, 0.5)
    assert ua.controller(compiled, "naive", "passive").output[0].level is None
    with pytest.raises(ValueError, match="base"):
        ua.controller(compiled, "projected")
    with pytest.raises(ValueError, match="correction"):
        ua.controller(compiled, "naive", "damped")
    with pytest.raises(ValueError, match="rest must be"):
        ua.controller(compiled, "frozen", rest="floating")
    with pytest.raises(ValueError, match="frozen controller"):
        ua.controller(compiled, "naive", rest="moving")
    with pytest.raises(ValueError, match="not both"):
        ua.controller(compiled, "frozen", rest="moving", at=[0.1, 0.2])


@pytest.mark.parametrize("gravity", [False, True])
@pytest.mark.parametrize("prefix", ROBOTS)
def test_force_tracking_against_the_lab(prefix, gravity):
    robot = build(prefix)
    compiled = law(prefix, robot, gravity, matrix=True)
    controller = ua.controller(compiled, "naive")
    lab_direction = to_3d(prefix, [1.0, 0.5])
    tracker = ua.DirectionalForce(controller, "tip", "ctrl.reach.stiffness", lab_direction, 3.0)
    tag = f"{prefix}_ft_g{int(gravity)}"
    for i in range(len(DATA[f"{prefix}_q"])):
        meas, q, v = reading(prefix, robot, i)
        controller.reset(0.0, meas)
        controller.step(0.0, meas)
        tracker.s = 5.0
        c0, cv = tracker._pieces(controller, q, v)
        np.testing.assert_allclose([c0, cv], DATA[f"{tag}_pieces"][i], rtol=1e-6, atol=1e-7)
        n = lab_direction / np.linalg.norm(lab_direction)
        goal = ua.direction_gain(c0, cv, 3.0, tracker.K, n)
        np.testing.assert_allclose(goal, DATA[f"{tag}_target"][i], rtol=1e-6, atol=1e-7)
        # the step moves s a fraction of the way, as the lab's low pass does (rate 0.2, dt 1e-2)
        tracker.s = 5.0
        tracker.step(controller, q, v, 1e-2)
        np.testing.assert_allclose(tracker.s, DATA[f"{tag}_step"][i], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(tracker.reading, c0 + tracker.s * cv, rtol=1e-12)
        read = c0 + 20.0 * cv  # the lab's reading at s = 20
        np.testing.assert_allclose(read * n, to_3d(prefix, DATA[f"{tag}_reading"][i]), atol=1e-8)


CLOSED_LOOPS = {
    "three": [("naive", "none", 0), ("frozen", "none", 0), ("naive", "tank", 0),
              ("frozen", "tank", 1), ("naive", "none", 1)],
    "five": [("naive", "none", 0)],
    "helyx": [("naive", "none", 0), ("frozen", "none", 0)],
}  # fmt: skip


HORIZON = {"three": 10, "five": 3, "helyx": 10}  # samples, 0.05 s apart, that stay comparable


@pytest.mark.parametrize("prefix", ROBOTS)
def test_closed_loops_follow_the_lab_simulation(prefix):
    """The lab integrates with RK4 and a continuous law; here the law is held for 0.2 ms, so the
    paths agree to the hold's first-order error (the five-link naive arm is nearly unstable and
    pulls them apart, hence its short horizon)."""
    robot = build(prefix)
    for base, correction, gravity in CLOSED_LOOPS[prefix]:
        tag = f"{prefix}_run_{base}_{correction}_g{gravity}"
        controller = lab_controller(
            law(prefix, robot, bool(gravity)), robot, base,
            None if correction == "none" else correction, bool(gravity),
        )  # fmt: skip
        q0 = np.full(robot.model.space.nq, {"three": 0.3, "five": 0.2, "helyx": 0.2}[prefix])
        plant = vmc.sim.ModelPlant(robot, q0=q0, max_step=1e-4)
        log = vmc.sim.run(plant, controller, vmc.sim.SimClock(2e-4), T=0.5)
        n = HORIZON[prefix]
        q = log.arrays()["q"][::250][:n]
        np.testing.assert_allclose(q, DATA[f"{tag}_q"][:n], atol=3e-3, err_msg=tag)
        if correction == "tank" and prefix == "three":  # the tank level follows the lab's
            assert controller.output[0].level == pytest.approx(DATA[f"{tag}_T"][10], rel=0.02)
