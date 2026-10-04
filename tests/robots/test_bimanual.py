"""The bimanual template: geometry, masses, calibration, a controller across both arms."""

import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.models import evaluate_frame
from virtualmodelcontrol.robots import bimanual, helyx


def test_straight_arms_stand_on_their_bases():
    robot = bimanual.arms()
    q = np.zeros(18)
    for arm, x in (("right", 0.125), ("left", -0.125)):
        np.testing.assert_allclose(
            evaluate_frame(robot.model, q, (arm, 1.0))[1], [x, 0, 0.58], atol=1e-15
        )
    masses = [float(robot.params[f"right_m{i}.mass"].value) for i in (1, 2, 3)]
    np.testing.assert_allclose(masses, [0.06, 0.03, 0.03])
    assert float(robot.params["right.efficiency.c1"].value) == 1.0  # the default efficiency


def test_each_arm_matches_the_single_arm_model():
    robot = bimanual.arms()
    single = helyx.arm("290-145-145").model
    rng = np.random.default_rng(14)
    q = rng.uniform(-0.01, 0.01, 18)
    for arm, sl, offset in (
        ("right", slice(0, 9), [0.125, 0, 0]),
        ("left", slice(9, 18), [-0.125, 0, 0]),
    ):
        for s in (0.3, 0.8, 1.0):
            np.testing.assert_allclose(
                evaluate_frame(robot.model, q, (arm, s))[1],
                evaluate_frame(single, q[sl], s)[1] + offset,
                atol=1e-15,
            )


def test_squeeze_controller_and_settling_simulation():
    robot = bimanual.add_dynamics(bimanual.arms())
    ctrl = vmc.Mechanism("ctrl")
    tips = robot.point("right", s=1.0) - robot.point("left", s=1.0)
    ctrl.add("squeeze", vmc.LinearSpring(tips - [0.15, 0.0, 0.0], 100.0))
    ctrl.add("gravity", vmc.GravityCompensation(robot))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    plant = vmc.sim.ModelPlant(robot)
    vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / bimanual.CONTROL_RATE), T=1.0)
    gap = (
        evaluate_frame(robot.model, plant.q, ("right", 1.0))[1]
        - evaluate_frame(robot.model, plant.q, ("left", 1.0))[1]
    )
    assert gap[0] < 0.25  # the tips moved towards each other
    assert np.all(np.isfinite(plant.q))
