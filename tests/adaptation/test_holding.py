from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import HoldingGoals
from virtualmodelcontrol.core.params import Param
from virtualmodelcontrol.estimation import ContactForce
from virtualmodelcontrol.robots import helyx

DATA = np.load(Path(__file__).parents[1] / "data" / "estimation.npz")
T_WB = DATA["t_wb"]  # the lab's arm base in its world frame; the library's arm sits at the origin
LAB = {"ctrl.s1.goal1": 0.5, "ctrl.s2.goal2": 1.0}
ARC = (0.5, 0.75, 1.0)
THREE = {f"ctrl.s{i}.g{i}": s for i, s in enumerate(ARC, 1)}  # three springs: as many as motors
DT = 1 / helyx.CONTROL_RATE
Q0 = 0.2 * np.array([0.05, 0.03, -0.02, 0.04, -0.03, 0.0, 0.02, 0.01, -0.02])


def lab_system():
    arm = helyx.arm("290-145-145", efficiency=float(DATA["eta"]))
    stiffness = Param("stiffness", DATA["k_struct"], unit="N/m", scope="design")
    arm.add("structure", vmc.LinearSpring(arm.joint(slice(0, arm.model.space.nq)), stiffness))
    ctrl = vmc.Mechanism("ctrl")
    for i, s in enumerate((0.5, 1.0), 1):
        ctrl.add(f"s{i}", vmc.LinearSpring(arm.point(s=s) - vmc.Ref(f"goal{i}", 3), np.eye(3)))
    return vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl), runtime=["ctrl.*"])


def three_springs(table=False, limit=None):
    """The soft arm held by three springs along it, one 3-vector goal for each three motors.

    With ``table`` the world has a ceiling just below the tip's pose at ``Q0``; with ``limit``
    the first goal may not go beyond it, in any direction."""
    arm = helyx.add_dynamics(helyx.arm("145-145-145"))
    if table:
        gap = vmc.PlaneDistance(arm.point(s=1.0), normal=[0, 0, -1], origin=[0, 0, 0.36])
        arm.add("table", vmc.ContactSpring(gap, 1000.0))
    ctrl = vmc.Mechanism("ctrl")
    for i, s in enumerate(ARC, 1):
        value = [0.0, 0.0, 0.1 * i]
        if limit is not None and i == 1:
            value = Param("g1", value, unit="m", scope="stage", bounds=(-limit, limit))
        goal = vmc.Ref(f"g{i}", 3, value=value)
        ctrl.add(f"s{i}", vmc.LinearSpring(arm.point(s=s) - goal, 200.0))
    ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    return arm, vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))


def start(compiled, arm):
    """A controller and a plant at rest at Q0, the controller's first step done."""
    controller, plant = vmc.VMCController(compiled), vmc.sim.ModelPlant(arm, q0=Q0)
    controller.reset(0.0, plant.read())
    controller.step(0.0, plant.read())
    return controller, plant


def run(controller, plant, seconds, repeat=None):
    """Simulate; ``repeat`` is a law and its targets, applied again every 0.1 s."""
    for k in range(int(seconds / DT)):
        plant.write(controller.step(plant.t, plant.read()))
        if repeat is not None and k % int(0.1 / DT) == 0:
            repeat[0].step(controller, repeat[1])
        plant.advance(DT)


def at(controller, theta):
    meas = vmc.Signals(0.0, motor_position=theta, motor_velocity=np.zeros(9))
    controller.reset(0.0, meas)
    controller.step(0.0, meas)


@pytest.mark.parametrize("i", range(3))
def test_the_goals_agree_with_the_labs_position_feedforward(i):
    compiled = lab_system()
    controller = vmc.VMCController(compiled)
    controller.set({f"ctrl.s{j + 1}.stiffness": DATA[f"hold/{i}/Kd"][j] for j in (0, 1)})
    at(controller, DATA[f"hold/{i}/q"])
    d0 = DATA[f"hold/{i}/d0"] - T_WB
    goals = HoldingGoals(controller, LAB).target(controller, dict(zip(LAB, d0, strict=True)))
    for j, name in enumerate(LAB):
        np.testing.assert_allclose(goals[name], DATA[f"hold/{i}/goals"][j] - T_WB, rtol=1e-9)


def test_the_goals_hold_the_arm_where_it_is():
    arm, compiled = three_springs()
    kin = vmc.Kinematics(arm)
    drift = {}
    for hold in (False, True):
        controller, plant = start(compiled, arm)
        tip = kin.position(plant.q, 1.0)
        if hold:
            law = HoldingGoals(controller, THREE)
            assert np.abs(ContactForce(controller, 1.0)(controller)).max() > 1.0  # pulled away
            law.step(controller)
            controller.step(0.0, plant.read())
            assert np.abs(ContactForce(controller, 1.0)(controller)).max() < 1e-9  # balanced
        run(controller, plant, 3.0)
        drift[hold] = np.linalg.norm(kin.position(plant.q, 1.0) - tip)
    assert drift[False] > 0.1  # the springs' own goals pull the tip away
    assert drift[True] < 1e-6


def test_goals_for_other_positions_bring_the_arm_near_them_and_again_closer():
    arm, compiled = three_springs()
    kin = vmc.Kinematics(arm)
    aim = {n: kin.position(1.3 * Q0 + 0.01, s) for n, s in THREE.items()}

    def worst(q):
        return max(np.linalg.norm(aim[n] - kin.position(q, s)) for n, s in THREE.items())

    ends = []
    for again in (False, True):
        controller, plant = start(compiled, arm)
        law = HoldingGoals(controller, THREE)
        law.step(controller, aim)
        run(controller, plant, 3.0, repeat=(law, aim) if again else None)
        ends.append(worst(plant.q))
    assert worst(Q0) > 0.3
    assert ends[0] < 0.05  # one feedforward, from the pose the arm had
    assert ends[1] < 0.3 * ends[0]  # the same law at the poses it reaches


def test_a_goal_that_is_not_live_is_refused():
    _, compiled = three_springs()
    with pytest.raises(ValueError, match="no live Param matches"):
        HoldingGoals(vmc.VMCController(compiled), {"ctrl.s1.nothing": 1.0})


def test_the_surroundings_the_model_leaves_out_do_not_enter_the_goals():
    bare, compiled = three_springs()
    world, in_contact = three_springs(table=True)
    free, _ = start(compiled, bare)
    touching, _ = start(in_contact, world)
    alone = HoldingGoals(free, THREE).target(free)
    left_out = HoldingGoals(touching, THREE, robot=bare).target(touching)
    counted = HoldingGoals(touching, THREE).target(touching)
    for name in THREE:
        np.testing.assert_allclose(left_out[name], alone[name], rtol=1e-9)
    assert max(np.abs(counted[n] - alone[n]).max() for n in THREE) > 1e-3


def test_the_goals_stay_within_the_bounds_of_their_params():
    arm, compiled = three_springs(limit=0.1)
    controller, _ = start(compiled, arm)
    law = HoldingGoals(controller, THREE)
    free = law.target(controller)
    assert np.abs(free["ctrl.s1.g1"]).max() > 0.1  # the holding goal is out of range
    law.step(controller)
    held = controller.live_params()
    assert np.abs(held["ctrl.s1.g1"]).max() == pytest.approx(0.1)
    np.testing.assert_allclose(held["ctrl.s2.g2"], free["ctrl.s2.g2"], rtol=1e-12)
