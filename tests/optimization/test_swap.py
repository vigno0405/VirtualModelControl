"""A planned swap, run on the library's simulator: the plan is what the swap executes."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from soft_arm_plan import HORIZON, NODES, SCALES, TRANSITION, soft_arm_swap
from virtualmodelcontrol import optimization as opt


def test_the_plan_is_what_the_swap_does_on_the_simulated_arm():
    arm, tip, goal, hold, new = soft_arm_swap()
    # Let the arm settle under the controller in place: the start of the plan is at rest.
    plant = vmc.sim.ModelPlant(arm)
    vmc.sim.run(plant, vmc.VMCController(vmc.compile(hold)), vmc.sim.SimClock(1 / 330), T=12.0)
    q0 = plant.q.copy()
    assert np.abs(plant.v).max() < 1e-9

    problem = opt.Problem(new)
    problem.add(
        opt.Collocation(q0, HORIZON, NODES, initial=hold, transition=TRANSITION, scales=SCALES)
    )
    problem.free("new.pull.stiffness")
    problem.add(opt.Effort(1e-4))
    problem.add(opt.Cost(tip - goal, t_from=TRANSITION, name="reach"))
    plan = problem.solve()
    assert plan.converged and plan.violation < 1e-4

    # Apply the plan to the new controller, swap to it over the planned transition, simulate.
    new_controller = vmc.VMCController(vmc.compile(new))
    new_controller.reset(0.0)
    plan.apply(new_controller)
    assert new_controller.live_params()["new.pull.stiffness"] == pytest.approx(
        plan.params["new.pull.stiffness"]
    )
    swap = vmc.control.SwapController(vmc.VMCController(vmc.compile(hold)))
    swap.swap(new_controller, TRANSITION)
    log = vmc.sim.run(vmc.sim.ModelPlant(arm, q0=q0), swap, vmc.sim.SimClock(1 / 330), T=HORIZON)
    rows = log.arrays()
    simulated = np.array(
        [np.interp(plan.t, rows["t"].ravel(), rows["q"][:, i]) for i in range(9)]
    ).T

    kin = vmc.Kinematics(arm)
    plan_tip = np.array([kin.position(q, 1.0) for q in plan.q])
    sim_tip = np.array([kin.position(q, 1.0) for q in simulated])
    travel = np.linalg.norm(plan_tip - plan_tip[0], axis=1).max()
    gap = np.linalg.norm(plan_tip - sim_tip, axis=1).max()
    assert travel > 0.05  # the tip really moves, about 8 cm
    assert gap < 0.05 * travel  # and the plan and the simulation agree to a few percent of it
    assert np.abs(plan.q - simulated).max() < 0.05 * np.abs(plan.q).max()
