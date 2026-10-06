from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.core.params import Param
from virtualmodelcontrol.estimation import ContactForce
from virtualmodelcontrol.robots import adapt, helyx

DATA = np.load(Path(__file__).parents[1] / "data" / "estimation.npz")
ETA = float(DATA["eta"])
T_WB = DATA["t_wb"]  # the arm's base in the lab's world frame; the library's arm sits at the origin
SITES = {"tip_normal": 1.0, "tip_free": 1.0, "mid_normal": 0.5}
CASES = [f"{name}/{i}" for name in SITES for i in range(3)]


def soft_arm(structure=True):
    arm = helyx.arm("290-145-145", efficiency=ETA)
    if structure:  # the arm's own stiffness, physical, in Δ
        stiffness = Param("stiffness", DATA["k_struct"], unit="N/m", scope="design")
        arm.add("structure", vmc.LinearSpring(arm.joint(slice(0, arm.model.space.nq)), stiffness))
    return arm


@pytest.fixture(scope="module")
def setup():
    arm = soft_arm()
    ctrl = vmc.Mechanism("ctrl")
    for i, s in enumerate((0.5, 1.0), 1):
        ctrl.add(f"s{i}", vmc.LinearSpring(arm.point(s=s) - vmc.Ref(f"goal{i}", 3), np.eye(3)))
    ctrl.add("damper", vmc.LinearDamper(arm.point(s=1.0), 5.0))  # pushes only while moving
    return vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl), runtime=["ctrl.*"])


def controller_at(compiled, key, velocity=0.0):
    controller = vmc.VMCController(compiled)
    values = {}
    for i in (0, 1):
        values[f"ctrl.s{i + 1}.stiffness"] = DATA[f"{key}/Kd"][i]
        values[f"ctrl.s{i + 1}.goal{i + 1}"] = DATA[f"{key}/d_ref"][i] - T_WB
    controller.set(values)
    meas = vmc.Signals(0.0, motor_position=DATA[f"{key}/q"], motor_velocity=np.full(9, velocity))
    controller.reset(0.0, meas)
    controller.step(0.0, meas)
    return controller


def estimator(controller, key, **kwargs):
    normal = DATA[f"{key}/normal"]
    return ContactForce(
        controller, SITES[key.split("/")[0]], None if not normal.any() else normal, **kwargs
    )


@pytest.mark.parametrize("key", CASES)
def test_the_force_agrees_with_the_labs_contact_force(setup, key):
    controller = controller_at(setup, key)
    force = estimator(controller, key)(controller)
    np.testing.assert_allclose(force, DATA[f"{key}/force"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(force, DATA[f"{key}/from_tau"], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("key", ["tip_normal/0", "mid_normal/1"])
def test_without_the_arms_stiffness_only_the_delivered_spring_term_is_left(setup, key):
    controller = controller_at(setup, key)
    bare = estimator(controller, key, robot=soft_arm(structure=False))(controller)
    np.testing.assert_allclose(bare, ETA * DATA[f"{key}/spring"], rtol=1e-9, atol=1e-12)
    held = DATA[f"{key}/force"] - bare  # what the arm's stiffness took
    np.testing.assert_allclose(held, -DATA[f"{key}/struct"], rtol=1e-9, atol=1e-12)


def test_velocities_are_left_out(setup):
    key = "tip_normal/2"
    still, moving = controller_at(setup, key), controller_at(setup, key, velocity=3.0)
    np.testing.assert_array_equal(estimator(still, key)(still), estimator(moving, key)(moving))


def test_the_force_follows_the_params_as_they_are_now(setup):
    key = "tip_free/1"
    controller = controller_at(setup, key)
    force = estimator(controller, key)
    before = force(controller)
    controller.set({"ctrl.s2.goal2": DATA[f"{key}/d_ref"][1] - T_WB + [0.0, 0.0, 0.01]})
    after = force(controller)
    assert np.abs(after - before).max() > 1e-3
    fresh = vmc.VMCController(setup)
    fresh.set(controller.live_params())
    meas = vmc.Signals(0.0, motor_position=DATA[f"{key}/q"], motor_velocity=np.zeros(9))
    fresh.reset(0.0, meas)
    fresh.step(0.0, meas)
    np.testing.assert_allclose(force(fresh), after, rtol=1e-12)


FINGER_TABLE = 1e4  # [N/m]


def test_a_fingertip_at_rest_on_a_table_reads_the_tables_force():
    model = adapt.add_dynamics(adapt.finger())  # the finger alone: the estimator's model
    world = adapt.add_dynamics(adapt.finger())  # the finger and its table: the simulated plant
    tip = world.point("tip")
    gap = vmc.PlaneDistance(tip, normal=[0, 0, -1], origin=[0, 0, 0.06])
    world.add("table", vmc.ContactSpring(gap, FINGER_TABLE))
    world.add("cushion", vmc.ContactDamper(gap, 5.0))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("press", vmc.LinearSpring(tip - vmc.Ref("goal", 3, value=[0, 0.05, 0.08]), 100.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 1.0))
    ctrl.add("gravity", vmc.GravityCompensation(world))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(world, ctrl)))
    plant = vmc.sim.ModelPlant(world, q0=[0.8, 0.8], max_step=1e-4)
    force = ContactForce(controller, "tip", [0, 0, 1], robot=model)
    kin = vmc.Kinematics(world)
    for _ in range(1500):
        plant.write(controller.step(plant.t, plant.read()))
        plant.advance(1 / 500)
    plant.write(controller.step(plant.t, plant.read()))
    true = FINGER_TABLE * (kin.position(plant.q, "tip")[2] - 0.06)
    assert true > 1.0  # pressing on it
    assert force(controller)[2] == pytest.approx(true, rel=0.02)
