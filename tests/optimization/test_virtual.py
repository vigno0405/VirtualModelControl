"""Virtual states in a plan: unknowns of the plan that the controller advances as it does."""

import copy
import itertools

import casadi as ca
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import virtualmodelcontrol as vmc
from small_plan import mass_spring
from virtualmodelcontrol import optimization as opt

LINK = "ctrl.link.stiffness"


def stateful(link=5.0, goal=1.0, anchor=3.0):
    """The mass of ``small_plan`` held by a controller with a virtual mass z: a spring from the
    mass to z, a spring and a damper from z to the goal, and a damper on the mass."""
    system, x, _ = mass_spring(goal=goal)
    ctrl = vmc.Mechanism("ctrl")
    z = ctrl.add_state("z", 1, unit="m")
    ctrl.add("mass", vmc.Inertance(z, 0.5))
    ctrl.add("link", vmc.LinearSpring(x - z, link))
    ctrl.add("anchor", vmc.LinearSpring(z - vmc.Ref("goal", 1, [goal]), anchor))
    ctrl.add("damper", vmc.LinearDamper(z, 1.0))
    ctrl.add("brake", vmc.LinearDamper(x, 0.5))
    return vmc.VirtualMechanismSystem(system.robot, ctrl), x


def stiffness(value=5.0):
    return vmc.Param("stiffness", value, bounds=(0.5, 50.0), scope="stage")


def plan_of(system, block, *terms, **options):
    problem = opt.Problem(system, **options)
    problem.add(block)
    for term in terms:
        problem.add(term)
    return problem, problem.solve()


@pytest.mark.parametrize(
    "integrator, substeps", list(itertools.product(("implicit", "rk4"), (1, 3)))
)
def test_a_shooting_with_virtual_states_is_the_closed_loop_of_the_simulator(integrator, substeps):
    system, _ = stateful()
    block = opt.Shooting([0.2], 3.0, 11, v0=[0.1], substeps=substeps, integrator=integrator)
    _, plan = plan_of(system, block)
    dt = 0.3 / substeps  # the last node too: the command of the state it ends in
    log = vmc.sim.rollout(system, [0.2], 3.3, dt, v0=[0.1], integrator=integrator, max_step=dt)
    assert plan.converged and plan.violation < 1e-9
    assert plan.z.shape == (11, 2) and np.abs(plan.z).max() > 0.5  # the state really moves
    for key in ("q", "v", "z", "u"):
        np.testing.assert_allclose(getattr(plan, key), log[key][::substeps], atol=1e-9)


def test_an_interval_dissipates_the_power_of_the_dampers_at_the_end_of_its_step():
    system, _ = stateful()
    problem, plan = plan_of(system, opt.Shooting([0.2], 2.0, 9, v0=[0.1]))
    nlp = problem.build()
    taken = ca.Function("taken", [nlp.x, nlp.p], nlp.trajectory.dissipated)
    taken = np.array(taken(problem.initial_guess(plan), np.zeros(0))).ravel()
    compiled, h = vmc.compile(system), 0.25
    for k in range(8):  # the damper on z takes power from the state's own velocity
        power = compiled.power(
            plan.q[k + 1], plan.v[k + 1], plan.z[k + 1], compiled.live_values(), (k + 1) * h
        )
        assert taken[k] == pytest.approx(-h * float(power[1]), rel=1e-9)
    assert taken.min() > 1e-3


def test_a_controller_without_virtual_states_has_none_in_the_plan():
    system, _, _ = mass_spring()
    _, plan = plan_of(system, opt.Shooting([0.0], 1.0, 5))
    assert plan.z.shape == (5, 0)
    _, plan = plan_of(system, opt.Collocation([0.0], 1.0, 5))
    assert plan.z.shape == (5, 0)


def test_a_running_controller_advances_its_state_over_the_first_step():
    system, _ = stateful()
    steps = 4  # the controller has run for four steps of the simulation
    log = vmc.sim.rollout(system, [0.2], 6.0, 0.3, v0=[0.1], max_step=0.3)
    start = {"v0": log["v"][steps], "z0": log["z"][steps]}
    _, running = plan_of(system, opt.Shooting(log["q"][steps], 3.0, 11, **start))
    for key in ("q", "v", "z", "u"):
        np.testing.assert_allclose(
            getattr(running, key)[:-1], log[key][steps : steps + 10], atol=1e-9
        )
    del start["z0"]  # started afresh, its state waits one step before it moves
    _, fresh = plan_of(system, opt.Shooting(log["q"][steps], 3.0, 11, **start))
    assert np.abs(fresh.q[:-1] - log["q"][steps : steps + 10]).max() > 1e-3


def test_a_controller_that_starts_with_the_plan_from_a_given_state_waits_one_step():
    system, _ = stateful()
    z0 = [0.3, 0.4]
    log = vmc.sim.rollout(system, [0.2], 3.3, 0.3, v0=[0.1], max_step=0.3, z0=z0)
    for kwargs, matches in (({"running": False}, True), ({}, False)):
        _, plan = plan_of(system, opt.Shooting([0.2], 3.0, 11, v0=[0.1], z0=z0, **kwargs))
        gap = np.abs(plan.q - log["q"]).max()
        assert bool(gap < 1e-9) == matches, (kwargs, gap)  # a controller said to be running moves
    _, plan = plan_of(system, opt.Shooting([0.2], 3.0, 11, v0=[0.1], z0=z0, running=True))
    assert np.abs(plan.q - log["q"]).max() > 1e-4


def test_the_state_the_plan_starts_from_is_the_one_given_or_the_compiled_one():
    system, _ = stateful()
    _, plan = plan_of(system, opt.Shooting([0.0], 1.0, 5))
    np.testing.assert_allclose(plan.z[0], [0.0, 0.0], atol=1e-12)
    _, plan = plan_of(system, opt.Shooting([0.0], 1.0, 5, z0=[0.3, -0.2]))
    np.testing.assert_allclose(plan.z[0], [0.3, -0.2], atol=1e-12)
    for block in (opt.Shooting([0.0], 1.0, 5, z0=[0.0]), opt.Collocation([0.0], 1.0, 5, z0=[0.0])):
        with pytest.raises(ValueError, match="positions, then velocities"):
            plan_of(system, block)
    _, plan = plan_of(system, opt.Collocation([0.0], 1.0, 9, z0=[0.3, -0.2]))
    np.testing.assert_allclose(plan.z[0], [0.3, -0.2], atol=1e-12)
    _, plan = plan_of(system, opt.Collocation([0.0], 1.0, 9))
    np.testing.assert_allclose(plan.z[0], [0.0, 0.0], atol=1e-12)


def test_the_state_of_a_shooting_is_a_parameter_of_the_program_when_it_is_given():
    system, _ = stateful()
    problem = opt.Problem(system)
    problem.add(opt.Shooting([0.0], 1.0, 5, z0=[0.0, 0.0]))
    problem.parameter("shooting.z0")
    first = problem.solve({"shooting.z0": [0.4, 0.1]})
    np.testing.assert_allclose(first.z[0], [0.4, 0.1], atol=1e-12)
    np.testing.assert_allclose(first.references["shooting.z0"], [0.4, 0.1])
    again = problem.solve({"shooting.z0": [-0.2, 0.0]}, warm_start=first)
    np.testing.assert_allclose(again.z[0], [-0.2, 0.0], atol=1e-12)


def exact_closed_loop(system, x0, horizon):
    """The closed loop in continuous time, [q, v, z, zdot], from SciPy's stiff solver."""
    sol = solve_ivp(
        vmc.sim.ode(system), (0.0, horizon), x0, method="Radau", rtol=1e-10, atol=1e-12,
        dense_output=True,
    )  # fmt: skip
    return sol.sol


def test_collocation_converges_to_the_continuous_closed_loop_at_the_order_of_its_scheme():
    system, _ = stateful()
    exact = exact_closed_loop(system, [0.2, 0.1, 0.0, 0.0], 3.0)
    errors = {}
    for scheme, nodes in (("trapezoid", 31), ("trapezoid", 61), ("hermite-simpson", 11),
                          ("hermite-simpson", 21)):  # fmt: skip
        _, plan = plan_of(system, opt.Collocation([0.2], 3.0, nodes, v0=[0.1], scheme=scheme))
        assert plan.converged and plan.violation < 1e-9
        reference = exact(plan.t).T
        errors.setdefault(scheme, []).append(
            (np.abs(plan.q[:, 0] - reference[:, 0]).max(), np.abs(plan.z - reference[:, 2:]).max())
        )
    for coarse, fine in zip(*errors["trapezoid"], strict=True):
        assert coarse / fine == pytest.approx(4.0, rel=0.1)  # halving the step quarters the error
    for coarse, fine in zip(*errors["hermite-simpson"], strict=True):
        assert coarse / fine == pytest.approx(16.0, rel=0.25)
    assert errors["hermite-simpson"][1][1] < 0.1 * errors["trapezoid"][1][1]  # the state's too


def test_the_virtual_states_of_a_collocation_have_their_own_constraints_and_unknowns():
    system, _ = stateful()
    plain, _, _ = mass_spring()
    for scheme in ("trapezoid", "hermite-simpson"):
        nlp = opt.Problem(system)
        nlp.add(opt.Collocation([0.0], 1.0, 11, scheme=scheme))
        base = opt.Problem(plain)
        base.add(opt.Collocation([0.0], 1.0, 11, scheme=scheme))
        built, bare = nlp.build(), base.build()
        assert built.x.numel() == bare.x.numel() + 22  # a position and a velocity per node
        rows = built.g.numel() - bare.g.numel()
        assert rows == 2 + 2 * 10  # the start, and the state's rule over every interval
        assert built.lbg[built.constraints["virtual"]].size == 20
        assert "virtual" not in bare.constraints


def test_a_periodic_motion_repeats_the_virtual_states_too():
    system, _ = stateful()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.5], 4.0, 21, periodic=True, v0=[0.0]))
    problem.add(opt.Cost(system.robot.joint(0) - 0.5, 1.0, name="stay"))
    plan = problem.solve(warm_start={"z": np.tile([0.5, 0.0], (21, 1))})
    assert plan.converged
    np.testing.assert_allclose(plan.z[-1], plan.z[0], atol=1e-8)
    np.testing.assert_allclose(plan.q[-1], plan.q[0], atol=1e-8)
    assert "virtual" in problem.build().constraints


def test_a_free_horizon_plans_the_virtual_states_over_the_time_it_finds():
    system, x = stateful()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 2.0, 21, free_time=(0.5, 5.0)))
    problem.add(
        opt.Cost(x - 1.0, 1.0)
    )  # a longer motion costs more: the horizon is the lower bound
    plan = problem.solve()
    reference = exact_closed_loop(system, [0.0, 0.0, 0.0, 0.0], plan.horizon)(plan.t).T
    assert plan.converged and plan.horizon == pytest.approx(0.5, abs=1e-6)
    np.testing.assert_allclose(plan.z, reference[:, 2:], atol=2e-2)


def test_the_terms_see_the_virtual_states_at_every_node_and_the_solver_starts_from_them():
    system, _ = stateful()
    start = [0.3, -0.2]
    for block, nodes in (
        (opt.Collocation([0.0], 1.0, 7, z0=start), 7),
        (opt.Equilibrium([0.0], z0=start), 1),
    ):
        problem = opt.Problem(system)
        problem.add(block)
        trajectory = problem.build().trajectory
        assert len(trajectory.z) == nodes and all(z.shape == (2, 1) for z in trajectory.z)
        x = problem.initial_guess()  # the solver's start: the state it was given at every node
        key = problem.build().variables.slices["z"]
        np.testing.assert_allclose(
            x[key].reshape(nodes, 2), np.tile([0.3, 0.0 if nodes == 1 else -0.2], (nodes, 1))
        )


def hold_system(system, stiffness=2.0):
    """A controller in place with a virtual state of its own, a mass that starts at 0.4 and is
    dragged to the goal 0."""
    mechanism = vmc.Mechanism("hold")
    x = system.robot.joint(0)
    z = mechanism.add_state("zh", 1, unit="m", initial=0.4)
    mechanism.add("mass", vmc.Inertance(z, 0.3))
    mechanism.add("link", vmc.LinearSpring(x - z, 6.0))
    mechanism.add("anchor", vmc.LinearSpring(z - 0.0, stiffness))
    mechanism.add("damper", vmc.LinearDamper(z, 0.8))
    mechanism.add("brake", vmc.LinearDamper(x, 0.5))
    return vmc.VirtualMechanismSystem(system.robot, mechanism)


def test_the_virtual_states_of_a_system_in_place_run_on_while_it_is_blended_out():
    system, _ = stateful()
    hold = hold_system(system)
    log = vmc.sim.rollout(hold, [0.2], 3.0, 0.3, v0=[0.1], max_step=0.3)
    slow = 1.0e4  # a transition so slow that the plan is the controller in place, all the way
    _, plan = plan_of(system, opt.Shooting([0.2], 3.0, 11, v0=[0.1], initial=hold, transition=slow))
    assert plan.converged and np.abs(log["z"]).max() > 0.3
    np.testing.assert_allclose(plan.q[:-1], log["q"], atol=1e-4)
    np.testing.assert_allclose(plan.u[:-1], log["u"], atol=1e-4)


def test_a_collocation_carries_the_state_of_a_running_controller_in_place_along():
    system, _ = stateful()
    hold = hold_system(system)
    running = vmc.VMCController(vmc.compile(hold))
    running.reset(0.0, vmc.Signals(0.0, motor_position=[0.5], motor_velocity=[0.0]))
    running.z = np.array([0.5, 0.0])  # at rest with the mass, but its anchor pulls it back
    exact = exact_closed_loop(hold, [0.5, 0.0, 0.5, 0.0], 3.0)
    for scheme, tolerance in (("trapezoid", 2e-3), ("hermite-simpson", 1e-4)):
        block = opt.Collocation([0.5], 3.0, 61, initial=running, transition=1.0e4, scheme=scheme)
        _, plan = plan_of(system, block)
        reference = exact(plan.t).T
        assert plan.converged and np.abs(reference[:, 0] - 0.5).max() > 0.1  # it moves
        assert np.abs(plan.q[:, 0] - reference[:, 0]).max() < tolerance


def test_a_running_controller_in_place_advances_its_own_state_over_the_first_step():
    system, _ = stateful()
    hold = hold_system(system)
    running = vmc.VMCController(vmc.compile(hold))
    h = 0.3
    for k in range(4):  # it has run for a while, somewhere else
        running.step(
            h * k, vmc.Signals(h * k, motor_position=[0.4 - 0.1 * k], motor_velocity=[0.2])
        )
    assert np.abs(running.z).max() > 1e-3
    twin = copy.deepcopy(running)
    _, plan = plan_of(
        system, opt.Shooting([0.2], 3.0, 11, v0=[0.1], initial=running, transition=1.0e4)
    )
    law = []
    for k in range(10):  # the plan's own motion, fed to the controller that goes on running
        meas = vmc.Signals(0.0, motor_position=plan.q[k], motor_velocity=plan.v[k])
        law.append(twin.step(h * (4 + k), meas)["law_torque"])
    assert plan.converged
    np.testing.assert_allclose(plan.u[:-1], np.array(law), atol=1e-4)


def test_a_system_in_place_waits_one_step_as_a_controller_after_a_reset_does():
    system, _ = stateful()
    hold = hold_system(system)
    twin = vmc.VMCController(vmc.compile(hold))
    _, plan = plan_of(
        system, opt.Shooting([0.2], 3.0, 11, v0=[0.1], initial=hold, transition=1.0e4)
    )
    law = []
    for k in range(10):
        meas = vmc.Signals(0.0, motor_position=plan.q[k], motor_velocity=plan.v[k])
        law.append(twin.step(0.3 * k, meas)["law_torque"])
    np.testing.assert_allclose(plan.u[:-1], np.array(law), atol=1e-4)


def test_an_equilibrium_of_the_closed_loop_rests_the_virtual_states_too():
    system, _ = stateful(goal=1.0)
    problem, plan = plan_of(system, opt.Equilibrium([0.2]))
    assert plan.converged and plan.violation < 1e-9
    np.testing.assert_allclose(plan.q, [[1.0]], atol=1e-8)
    np.testing.assert_allclose(plan.z, [[1.0, 0.0]], atol=1e-8)  # at the goal, not moving
    np.testing.assert_allclose(plan.u, [[0.0]], atol=1e-8)
    assert "virtual" in problem.build().constraints
    start = plan_of(system, opt.Equilibrium([0.2], z0=[0.7, 0.0]))[1]
    np.testing.assert_allclose(start.z, [[1.0, 0.0]], atol=1e-8)
    with pytest.raises(ValueError, match="positions, then velocities"):
        plan_of(system, opt.Equilibrium([0.2], z0=[0.7]))


def test_the_equilibrium_balances_the_forces_on_the_virtual_state_as_well():
    system, _ = stateful(goal=1.0, anchor=3.0)
    system.virtual.add("push", vmc.ForceSource(system.virtual.states["z"], [0.6]))
    plan = plan_of(system, opt.Equilibrium([0.2]))[1]
    assert plan.converged
    np.testing.assert_allclose(plan.z, [[1.2, 0.0]], atol=1e-8)  # 3 (z - 1) = 0.6
    np.testing.assert_allclose(plan.q, [[1.2]], atol=1e-8)  # and the mass sits with it


def tank_program(level):
    """The mass pulled to 1 through a virtual mass whose link stiffness steps, a tank pays."""
    system, x = stateful(link=stiffness())
    problem = opt.Problem(system, solver="ipopt-exact")
    problem.add(opt.Shooting([0.0], 1.6, 9, steps=[LINK], substeps=4))
    problem.add(opt.Effort(0.01))
    problem.add(opt.Cost(x - 1.0, 1.0, name="reach"))
    problem.add(opt.TankBudget(level))
    return system, problem


def test_the_planned_level_of_the_tank_uses_the_energy_jumps_at_the_virtual_states_of_the_plan():
    system, problem = tank_program(0.05)
    plan = problem.solve()
    nlp = problem.build()
    x = problem.initial_guess(plan)
    taken = np.array(
        ca.Function("taken", [nlp.x, nlp.p], nlp.trajectory.dissipated)(x, np.zeros(0))
    )
    controller = vmc.VMCController(vmc.compile(system, runtime=[LINK]))
    levels, wrong, right, level = [], [], [], 0.05
    for k in range(8):  # the controller at the plan's node k, given the plan's step of its Param
        meas = vmc.Signals(0.0, motor_position=plan.q[k], motor_velocity=plan.v[k])
        controller.reset(0.0, meas, z0=np.zeros(2))
        wrong.append(controller.jump({LINK: plan.steps[LINK][k]}))
        controller.reset(0.0, meas, z0=plan.z[k])
        jump = controller.set({LINK: plan.steps[LINK][k]})
        level = (level + (taken.ravel()[k - 1] if k else 0.0)) - jump
        right.append(jump)
        levels.append(level)
    x[nlp.variables.slices["level:tank"]] = levels
    g = np.array(nlp.functions()[1](x, np.zeros(0))).ravel()
    assert plan.converged and np.abs(g[nlp.constraints["tank"]]).max() < 1e-7
    assert np.abs(np.array(wrong) - np.array(right)).max() > 1e-4  # the state of z matters


def test_the_warm_start_takes_the_virtual_states_and_checks_their_shape():
    system, _ = stateful()
    problem, plan = plan_of(system, opt.Collocation([0.0], 2.0, 21))
    again = problem.solve(warm_start=plan)
    assert again.iterations <= 1 and again.converged
    with pytest.raises(ValueError, match="warm start 'z'"):
        problem.solve(warm_start={"z": np.zeros((21, 3))})
    np.testing.assert_allclose(
        problem.initial_guess(plan),
        problem.initial_guess({"z": plan.z, "q": plan.q, "v": plan.v, "a": plan.a}),
    )


def mpc_program(z0=True):
    system, x = stateful(link=stiffness())
    problem = opt.Problem(system, solver="ipopt-exact")
    problem.add(
        opt.Shooting([0.0], 1.0, 11, steps=[LINK], substeps=5, z0=[0.0, 0.0] if z0 else None)
    )
    problem.add(opt.Effort(0.01))
    problem.add(opt.Cost(x - 1.0, 1.0, name="reach"))
    return system, problem


def test_an_mpc_plans_from_the_state_of_the_running_controller_and_shifts_it():
    system, problem = mpc_program()
    mpc = opt.MPC(problem)
    assert "shooting.z0" in mpc.nlp.parameters
    controller = vmc.VMCController(vmc.compile(system, runtime=[LINK]))
    controller.reset(0.0, vmc.Signals(0.0, motor_position=[0.0], motor_velocity=[0.0]))
    for k in range(3):
        controller.step(0.1 * k, vmc.Signals(0.0, motor_position=[0.1 * k], motor_velocity=[0.5]))
        z = controller.z.copy()
        mpc.step(controller, [0.1 * k], [0.5])
        np.testing.assert_allclose(mpc.result.z[0], z, atol=1e-12)
        assert mpc.result.converged
    assert mpc._warm["z"].shape == (11, 2)
    np.testing.assert_allclose(mpc._warm["z"][0], mpc.result.z[1])  # one interval on


def test_the_first_solve_of_an_mpc_starts_from_the_state_of_the_controller():
    system, problem = mpc_program()
    mpc = opt.MPC(problem)
    controller = vmc.VMCController(vmc.compile(system, runtime=[LINK]))
    controller.reset(0.0, vmc.Signals(0.0, motor_position=[0.0], motor_velocity=[0.0]))
    controller.z = np.array([0.4, -0.1])
    warm, solve = [], problem.solve
    problem.solve = lambda refs, warm_start=None: (
        warm.append(warm_start),
        solve(refs, warm_start=warm_start),
    )[1]
    mpc.step(controller, [0.0], [0.0])
    np.testing.assert_allclose(warm[0]["z"], np.tile([0.4, -0.1], (11, 1)))


def test_an_mpc_of_a_controller_with_virtual_states_needs_the_state_it_runs_from():
    _, problem = mpc_program(z0=False)
    with pytest.raises(ValueError, match="virtual states"):
        opt.MPC(problem)
