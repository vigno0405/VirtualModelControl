"""The momentum observer: a first-order lag of the true external force, exact model terms."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.estimation import MomentumObserver
from virtualmodelcontrol.models import JointSpace, SerialChain
from virtualmodelcontrol.robots import adapt

DT, K = 1 / 500, 40.0  # [s], [1/s]


def idle(robot):
    return vmc.VirtualMechanismSystem(robot, vmc.Mechanism("idle"))


def run(plant, observer, controller=None, seconds=2.0):
    """Step the plant under ``controller`` (no command by default) and the observer at 500 Hz;
    rows of (t, estimate, true external generalized force of the element 'world')."""
    rows = []
    for _ in range(int(seconds / DT)):
        u = np.zeros(plant.dynamics.n_u)
        if controller is not None:
            u = controller.step(plant.t, plant.read())["motor_torque"]
            plant.write(vmc.Signals(plant.t, motor_torque=u))
        estimate = observer(plant.q, plant.v, u, plant.t)
        true = plant.elements()["world"]["torque"]
        rows.append((plant.t, estimate.copy(), true.copy()))
        plant.advance(DT)
    return rows


def mass_with_world(stiffness=50.0, mass=2.0):
    """A mass on a slider. The 'world' (a spring to the origin) is only in the plant's copy."""
    bare = vmc.Mechanism("mass", model=JointSpace(1, unit="m"))
    bare.add("m", vmc.Inertance(bare.joint(0), mass))
    plant = vmc.Mechanism("mass", model=JointSpace(1, unit="m"))
    plant.add("m", vmc.Inertance(plant.joint(0), mass))
    plant.add("world", vmc.LinearSpring(plant.joint(0), stiffness))
    return bare, plant


def test_the_estimate_is_a_first_order_lag_of_the_true_force():
    bare, world = mass_with_world()
    observer = MomentumObserver(idle(bare), DT, K)
    plant = vmc.sim.ModelPlant(world, q0=[0.1], max_step=1e-4)  # released 10 cm from the origin
    t, estimate, true = (np.array(x) for x in zip(*run(plant, observer), strict=True))
    lag = np.zeros_like(true)  # the lag the observer should be: r' = K (r_true - r)
    for k in range(1, len(t)):
        lag[k] = lag[k - 1] + DT * K * (true[k - 1] - lag[k - 1])
    assert np.abs(true).max() == pytest.approx(5.0, rel=1e-6)  # 50 N/m x 0.1 m at the start
    assert np.abs(estimate - lag).max() < 0.03 * np.abs(true).max()
    assert np.abs(estimate - true).max() > 0.2 * np.abs(true).max()  # it does lag, at 40 /s


def test_without_surroundings_a_moving_robot_shows_no_force_to_the_exact_model():
    finger = adapt.add_dynamics(adapt.finger())
    tip = finger.point("tip")
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("pull", vmc.LinearSpring(tip - [0.0, 0.05, 0.06], 100.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 1.0))
    ctrl.add("limits", adapt.joint_limit_spring(finger))
    ctrl.add("gravity", vmc.GravityCompensation(finger))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(finger, ctrl)))
    observer = MomentumObserver(controller.compiled.system, DT, K)
    plant = vmc.sim.ModelPlant(finger, q0=[0.4, 1.2], max_step=1e-4)
    still = [plant.elements()]  # the finger has no 'world'; look at what it feels instead
    assert "world" not in still[0]
    biggest, moved = 0.0, 0.0
    for _ in range(500):  # a second of motion
        u = controller.step(plant.t, plant.read())["motor_torque"]
        plant.write(vmc.Signals(plant.t, motor_torque=u))
        moved = max(moved, float(np.abs(plant.v).max()))
        biggest = max(biggest, float(np.abs(observer(plant.q, plant.v, u, plant.t)).max()))
        plant.advance(DT)
    assert moved > 0.5  # [rad/s]: it moves
    assert biggest < 2e-3  # [N·m]: and the observer finds no force on it, Coriolis included


def table_world(table=1e4):
    plant = adapt.add_dynamics(adapt.finger())
    tip = plant.point("tip")
    gap = vmc.PlaneDistance(tip, normal=[0, 0, -1], origin=[0, 0, 0.06])
    plant.add("world", vmc.ContactSpring(gap, table))
    plant.add("cushion", vmc.ContactDamper(gap, 5.0))
    return plant


def test_the_force_of_a_table_is_found_from_the_model_without_it():
    plant_robot = table_world()
    model = adapt.add_dynamics(adapt.finger())
    tip = plant_robot.point("tip")
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("press", vmc.LinearSpring(tip - [0.0, 0.05, 0.075], 100.0))  # 1.5 cm into it
    ctrl.add("damp", vmc.LinearDamper(tip, 1.0))
    ctrl.add("limits", adapt.joint_limit_spring(plant_robot))
    ctrl.add("gravity", vmc.GravityCompensation(plant_robot))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(plant_robot, ctrl)))
    observer = MomentumObserver(controller.compiled.system, DT, K, robot=model)
    plant = vmc.sim.ModelPlant(plant_robot, q0=[0.8, 0.8], max_step=1e-4)
    rows = run(plant, observer, controller, seconds=3.0)
    t, estimate, true = (np.array(x) for x in zip(*rows, strict=True))
    late = t > 2.5
    assert np.abs(true[late]).max() > 0.02  # [N·m] it presses
    np.testing.assert_allclose(estimate[late], true[late], rtol=0.02, atol=1e-4)  # the torques
    force = observer.force("tip", normal=[0, 0, 1])  # the force on the table [N]
    kin = vmc.Kinematics(plant_robot)
    below = kin.position(plant.q, "tip")[2] - 0.06  # [m] into the table, as the observer saw it
    assert force[2] == pytest.approx(1e4 * max(0.0, below), rel=0.05)


def test_a_fast_free_swing_shows_no_force_and_a_late_start_begins_at_none():
    finger = adapt.add_dynamics(adapt.finger())
    plant = vmc.sim.ModelPlant(finger, q0=[0.4, 1.2], v0=[3.0, -4.0], max_step=1e-4)
    observer = MomentumObserver(idle(finger), DT, K)
    nothing = np.zeros(plant.dynamics.n_u)
    plant.advance(0.1)  # the finger is well under way when the observer starts
    assert np.abs(plant.v).max() > 5.0  # [rad/s]
    assert np.abs(observer(plant.q, plant.v, nothing, plant.t)).max() == 0.0  # it starts at none
    biggest = 0.0
    for _ in range(250):
        plant.advance(DT)
        r = observer(plant.q, plant.v, nothing, plant.t)
        biggest = max(biggest, float(np.abs(r).max()))
    assert biggest < 5e-4  # [N·m] against a Coriolis torque of tenths, with the exact model


def test_reset_forgets_the_estimate():
    finger = adapt.add_dynamics(adapt.finger())
    observer = MomentumObserver(idle(finger), DT, K)
    nothing = np.zeros(2)
    observer([0.4, 1.2], [3.0, -4.0], nothing)
    observer([0.4, 1.2], [3.0, -4.0], nothing)
    observer.reset()
    assert np.abs(observer([0.1, 0.3], [-2.0, 1.0], nothing)).max() == 0.0


def test_the_force_is_along_the_normal_asked_for():
    plant_robot = table_world()
    observer = MomentumObserver(idle(plant_robot), DT, K, robot=adapt.add_dynamics(adapt.finger()))
    plant = vmc.sim.ModelPlant(plant_robot, q0=[0.8, 0.8], max_step=1e-4)
    for _ in range(100):
        observer(plant.q, plant.v, np.zeros(2), plant.t)
        plant.advance(DT)
    along = observer.force("tip", normal=[0, 0, 1])
    assert np.abs(along[:2]).max() < 1e-12 and abs(along[2]) > 1e-3
    assert abs(observer.force("tip")[1]) > 1e-6  # without a normal, the whole vector


def test_a_floating_body_is_refused():
    chain = SerialChain(["floating"], axes=[None], points=[[0, 0, 0]], sites={"c": (1, [0, 0, 0])})
    body = vmc.Mechanism("body", model=chain)
    body.add("m", vmc.PointMass(body.point("c"), 1.0))
    with pytest.raises(ValueError, match="velocity as configuration"):
        MomentumObserver(idle(body), DT, K)


def test_the_model_is_read_at_its_live_values():
    def with_spring():
        robot = vmc.Mechanism("mass", model=JointSpace(1, unit="m"))
        robot.add("m", vmc.Inertance(robot.joint(0), 2.0))
        spring = vmc.LinearSpring(robot.joint(0), 50.0)
        robot.add("own", spring)
        return robot, spring

    model, spring = with_spring()
    plant = vmc.sim.ModelPlant(with_spring()[0], q0=[0.1], max_step=1e-4)
    observer = MomentumObserver(idle(model), DT, K)
    nothing = np.zeros(1)
    for _ in range(100):
        assert abs(observer(plant.q, plant.v, nothing, plant.t)).max() < 0.1  # [N] of 5
        plant.advance(DT)
    spring.stiffness.value = 80.0  # the model now believes a stiffer spring than the plant has
    estimate = [observer(plant.q, plant.v, nothing, plant.t) for _ in range(50)]
    assert abs(estimate[-1][0]) > 0.5  # [N]: the 30 N/m it adds, at about 0.1 m, lagged by 1/K
