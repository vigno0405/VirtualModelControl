"""Laws and estimators that read the motors refuse a controller that reads the state."""

import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import (
    ForceTracking,
    HoldingGoals,
    PositionRegulation,
    StiffnessTracking,
)
from virtualmodelcontrol.control import StateController, Tank
from virtualmodelcontrol.estimation import ContactForce, TaskStiffness
from virtualmodelcontrol.robots import planar

K, GOAL = "ctrl.reach.stiffness", "ctrl.reach.goal"


def compiled():
    """The three-link arm, held by a spring whose stiffness and goal are live."""
    arm = planar.add_dynamics(planar.arm("three-link"))
    stiffness = vmc.Param("stiffness", 150.0, bounds=(1.0, 1000.0), scope="stage")
    goal = vmc.Ref("goal", 3, vmc.Param("goal", [0.45, 0.15, 0.0], bounds=(-1, 1), scope="stage"))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(arm.point("tip") - goal, stiffness))
    return vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))


BUILDERS = {
    "ForceTracking": lambda c: ForceTracking(c, "tip", K),
    "StiffnessTracking": lambda c: StiffnessTracking(c, "tip", K),
    "PositionRegulation": lambda c: PositionRegulation(c, {GOAL: "tip"}),
    "HoldingGoals": lambda c: HoldingGoals(c, {GOAL: "tip"}),
    "ContactForce": lambda c: ContactForce(c, "tip"),
    "TaskStiffness": lambda c: TaskStiffness(c, "tip"),
}


@pytest.mark.parametrize("name", list(BUILDERS))
def test_a_law_or_an_estimator_on_the_motors_refuses_a_state_controller(name):
    system = compiled()
    build = BUILDERS[name]
    motors, state = vmc.VMCController(system), StateController(system)
    assert build(motors) is not None  # a controller that reads the motors is accepted
    for refused in (state, Tank(state, level=1.0)):
        with pytest.raises(ValueError, match=f"{name} reads the motors.*StateController"):
            build(refused)
    assert build(Tank(motors, level=1.0)) is not None  # a tank around it does not hide that
