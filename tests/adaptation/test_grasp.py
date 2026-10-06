"""Two arms hold an object, each arm tracking its own force along the line between the tips."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import ForceTracking
from virtualmodelcontrol.estimation import ContactForce
from virtualmodelcontrol.robots import bimanual

WIDTH, K_OBJECT, WANTED = 0.10, 200.0, 1.0  # [m], [N/m], [N]
NORMALS = {"right": np.array([-1.0, 0.0, 0.0]), "left": np.array([1.0, 0.0, 0.0])}


@pytest.fixture(scope="module")
def world():
    arms = bimanual.add_dynamics(bimanual.arms())
    tips = {arm: arms.point(arm, s=1.0) for arm in bimanual.ARMS}
    arms.add("object", vmc.ContactSpring(vmc.Norm(tips["right"] - tips["left"]) - WIDTH, K_OBJECT))
    ctrl = vmc.Mechanism("ctrl")
    for arm, x in (("right", 0.05), ("left", -0.05)):  # the goals start on the object's faces
        goal = vmc.Ref(f"goal_{arm}", 3, value=[x, 0.0, 0.58])
        ctrl.add(arm, vmc.LinearSpring(tips[arm] - goal, 30.0))
        ctrl.add(f"damp_{arm}", vmc.LinearDamper(tips[arm], 2.0))
    ctrl.add("gravity", vmc.GravityCompensation(arms))
    return arms, vmc.compile(vmc.VirtualMechanismSystem(arms, ctrl))


def estimated(controller):
    """What each arm believes, from the model of the arms alone (no object)."""
    model = bimanual.add_dynamics(bimanual.arms())
    force = {
        arm: ContactForce(controller, (arm, 1.0), NORMALS[arm], robot=model)
        for arm in bimanual.ARMS
    }
    return lambda arm, true: force[arm](controller)


def measured(controller):
    """What a sensor on the object reads: the object's force along each arm's normal."""
    return lambda arm, true: true * NORMALS[arm]


def hold(world, make_sensor, seconds=5.0, level=None):
    """Run with each law told ``sensor(arm, true_force)`` from 1.5 s on, through a tank with
    ``level`` joules if given."""
    arms, compiled = world
    controller = vmc.VMCController(compiled)
    if level is not None:
        controller = vmc.control.Tank(controller, level=level)
    sensor = make_sensor(controller)
    laws = {
        arm: ForceTracking(controller, (arm, 1.0), f"ctrl.{arm}.goal_{arm}", NORMALS[arm])
        for arm in bimanual.ARMS
    }
    plant = vmc.sim.ModelPlant(arms)
    kin = vmc.Kinematics(arms)
    log = []
    dt = 1 / bimanual.CONTROL_RATE
    for _ in range(int(seconds / dt)):
        plant.write(controller.step(plant.t, plant.read()))
        tips = [kin.position(plant.q, (arm, 1.0)) for arm in bimanual.ARMS]
        true = K_OBJECT * max(0.0, WIDTH - np.linalg.norm(tips[0] - tips[1]))
        told = {arm: sensor(arm, true) for arm in bimanual.ARMS}
        if plant.t > 1.5:
            for arm, law in laws.items():
                law.step(controller, told[arm], WANTED * NORMALS[arm])
        log.append((plant.t, true, *(told[arm] @ NORMALS[arm] for arm in bimanual.ARMS)))
        plant.advance(dt)
    return np.array(log)


def test_each_arm_tracks_its_force_from_the_model_alone(world):
    log = hold(world, estimated)
    late = log[-500:]
    assert late[:, 1].mean() == pytest.approx(WANTED, abs=0.005)  # the object's true force
    for column in (2, 3):  # what each arm believed
        assert late[:, column].mean() == pytest.approx(WANTED, abs=0.02)
    assert np.ptp(late[:, 1]) < 0.05


def test_each_arm_tracks_the_force_a_sensor_on_the_object_gives(world):
    log = hold(world, measured)
    assert log[-500:, 1].mean() == pytest.approx(WANTED, abs=0.06)
    assert log[:700, 1].max() < 0.01  # nothing before the arms touch it and the laws start


def test_through_a_tank_the_grasp_is_paid_for_by_its_energy(world):
    empty = hold(world, estimated, level=0.0)[-500:, 1].mean()
    funded = hold(world, estimated, level=0.2)[-500:, 1].mean()
    assert empty < 0.5 * WANTED  # the goals can move only as far as the arms' dampers paid
    assert funded == pytest.approx(WANTED, abs=0.02)


def test_with_a_sensor_on_the_object_the_tank_also_decides_how_far_the_grasp_gets(world):
    empty = hold(world, measured, level=0.0)[-500:, 1].mean()
    funded = hold(world, measured, level=0.2)[-500:, 1].mean()
    assert empty < 0.5 * WANTED
    assert funded == pytest.approx(WANTED, abs=0.06)
