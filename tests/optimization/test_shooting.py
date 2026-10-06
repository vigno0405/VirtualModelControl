"""Multiple shooting: the closed loop of the simulator, joined by constraints, and its Params."""

import itertools

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import virtualmodelcontrol as vmc
from small_plan import DAMPING, exact_position, mass_spring, tanh_mass, two_masses
from virtualmodelcontrol import optimization as opt


def swing(stiffness=8.0, goal=1.0, controlled=True):
    """A pendulum under gravity; a spring-damper controller to ``goal``, or an idle one."""
    chain = vmc.models.SerialChain(
        ["revolute"], axes=[[0, 1, 0]], points=[[0, 0, 0]], sites={"bob": (1, [0.5, 0, 0])}
    )
    robot = vmc.Mechanism("robot", model=chain, actuation=vmc.models.Direct(1.0))
    robot.add_param(vmc.Param("gravity", [0.0, 0.0, -9.81], unit="m/s^2"))
    robot.add("bob", vmc.PointMass(robot.point("bob"), 1.0))
    robot.add("gravity", vmc.Gravity(robot))
    q = robot.joint(0)
    robot.add("friction", vmc.LinearDamper(q, 0.2))
    ctrl = vmc.Mechanism("ctrl")
    if controlled:
        k = vmc.Param("stiffness", stiffness, bounds=(0.5, 60.0), scope="stage")
        ctrl.add("spring", vmc.LinearSpring(q - vmc.Ref("goal", 1, [goal]), k))
        ctrl.add("damper", vmc.LinearDamper(q, 1.0))
    return vmc.VirtualMechanismSystem(robot, ctrl), q


def plan_of(system, block, *terms, **options):
    problem = opt.Problem(system, **options)
    problem.add(block)
    for term in terms:
        problem.add(term)
    return problem, problem.solve()


def test_a_shooting_is_the_closed_loop_of_the_simulator_step_for_step():
    for integrator in ("implicit", "rk4"):
        system, _, _ = mass_spring()
        block = opt.Shooting([0.2], 3.0, 11, v0=[0.1], substeps=5, integrator=integrator)
        _, plan = plan_of(system, block)
        # 5 control steps of 0.06 s per interval, the command held over each
        log = vmc.sim.rollout(system, [0.2], 3.0, 0.06, v0=[0.1], integrator=integrator, max_step=0.06)
        assert plan.converged and plan.violation < 1e-9
        np.testing.assert_allclose(plan.q[:-1], log["q"][::5], atol=1e-9)
        np.testing.assert_allclose(plan.v[:-1], log["v"][::5], atol=1e-9)
        np.testing.assert_allclose(plan.u[:-1], log["u"][::5], atol=1e-9)


def test_the_error_against_the_exact_solution_halves_with_every_doubling_of_the_control_steps():
    errors = []
    for substeps in (4, 8, 16):
        system, _, _ = mass_spring()
        _, plan = plan_of(system, opt.Shooting([0.0], 3.0, 11, substeps=substeps))
        errors.append(np.abs(plan.q[:, 0] - exact_position(plan.t)).max())
    for coarse, fine in itertools.pairwise(errors):
        assert coarse / fine == pytest.approx(2.0, rel=0.15)  # the held command: first order


def ode_error(integrator, substeps):
    """The idle pendulum's largest error at the nodes, against SciPy's solution."""
    system, _ = swing(controlled=False)
    _, plan = plan_of(system, opt.Shooting([1.0], 2.0, 5, substeps=substeps, integrator=integrator))
    f = vmc.sim.ode(system)
    truth = solve_ivp(f, (0.0, 2.0), [1.0, 0.0], t_eval=plan.t, rtol=1e-11, atol=1e-12)
    return np.abs(plan.q[:, 0] - truth.y[0]).max()


def test_rk4_is_fourth_order_and_the_implicit_step_first_order_on_a_nonlinear_robot():
    rk4 = [ode_error("rk4", s) for s in (4, 8, 16)]
    implicit = [ode_error("implicit", s) for s in (16, 32, 64)]
    for coarse, fine in itertools.pairwise(rk4):
        assert coarse / fine == pytest.approx(16.0, rel=0.1)
    for coarse, fine in itertools.pairwise(implicit):
        assert coarse / fine == pytest.approx(2.0, rel=0.2)
    assert rk4[0] < implicit[-1] / 10  # RK4 with 4 steps beats the implicit step with 64


def test_the_plan_agrees_with_collocation_on_the_pendulum_and_the_gap_closes_with_substeps():
    def best(block):
        system, q = swing()
        problem = opt.Problem(system, solver="ipopt-exact")
        problem.add(block)
        problem.free("ctrl.spring.stiffness")
        problem.add(opt.Effort(0.02))
        problem.add(opt.Cost(q - 1.0, 1.0, name="reach"))
        return problem.solve()

    reference = best(opt.Collocation([0.0], 2.0, 11, scheme="hermite-simpson"))
    assert reference.converged
    cost, path = [], []
    for substeps in (8, 32):
        plan = best(opt.Shooting([0.0], 2.0, 11, substeps=substeps))
        assert plan.converged
        cost.append(abs(plan.cost / reference.cost - 1.0))
        path.append(np.abs(plan.q[:, 0] - reference.q[:, 0]).max())  # the same nodes
    k = "ctrl.spring.stiffness"
    assert cost[1] < cost[0] and path[1] < path[0]
    assert cost[1] < 0.02 and path[1] < 0.02
    assert plan.params[k].item() == pytest.approx(reference.params[k].item(), rel=0.05)


def test_the_start_is_a_parameter_of_one_program():
    system, x, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Shooting([0.0], 2.0, 11, substeps=4))
    problem.parameter("shooting.q0", "shooting.v0")
    problem.build()
    a = problem.solve({"shooting.q0": [0.5], "shooting.v0": [-0.3]})
    b = problem.solve({"shooting.q0": [-0.2]})
    assert (a.q[0, 0], a.v[0, 0], b.q[0, 0], b.v[0, 0]) == pytest.approx((0.5, -0.3, -0.2, 0.0))
    assert problem.build() is problem.build()
    fixed = plan_of(system, opt.Shooting([0.5], 2.0, 11, v0=[-0.3], substeps=4))[1]
    np.testing.assert_allclose(a.q, fixed.q, atol=1e-9)


def test_the_continuity_holds_with_steps_that_the_simulator_applies_at_the_same_times():
    system, x = tanh_mass(stiffness=2.0, max_force=5.0, goal=1.0, bounds=(0.5, 50.0))
    k, goal = "ctrl.spring.stiffness", "ctrl.spring.goal"
    block = opt.Shooting([0.0], 1.0, 6, steps=[k, goal], substeps=4)
    problem = opt.Problem(system)
    problem.add(block)
    stiffness, goals = [3.0, 20.0, 8.0, 40.0, 1.0], [0.4, 1.0, 0.2, 0.8, 0.0]
    plant = vmc.sim.ModelPlant(system.robot, q0=[0.0], max_step=0.05)  # one step per control step
    controller = vmc.VMCController(vmc.compile(system))
    controller.reset(0.0, plant.read())
    nodes = []
    for step in range(20):  # 4 control steps of 0.05 s in each interval of 0.2 s
        if step % 4 == 0:
            nodes.append((plant.q.copy(), plant.v.copy()))
            controller.set({k: stiffness[step // 4], goal: [goals[step // 4]]})
        plant.write(controller.step(plant.t, plant.read()))
        plant.advance(0.05)
    nodes.append((plant.q.copy(), plant.v.copy()))
    steps = {k: np.array(stiffness), goal: np.array(goals)[:, None]}
    warm = {"q": [n[0] for n in nodes], "v": [n[1] for n in nodes], "steps": steps}
    nlp = problem.build()
    x_sim = problem.initial_guess(warm)
    g = np.array(nlp.functions()[1](x_sim, np.zeros(0))).ravel()
    assert np.abs(g[nlp.constraints["continuity"]]).max() < 1e-9
    assert np.abs(g[nlp.constraints["start"]]).max() == 0.0
    shifted = {**steps, k: np.array(stiffness)[::-1]}  # other values: the same nodes break it
    x_bad = problem.initial_guess({**warm, "steps": shifted})
    g_bad = np.array(nlp.functions()[1](x_bad, np.zeros(0))).ravel()
    assert np.abs(g_bad[nlp.constraints["continuity"]]).max() > 1e-3


def test_the_steps_come_back_per_interval_in_the_shape_of_the_param_and_apply_one_of_them():
    system, q = two_masses()
    problem = opt.Problem(system)
    problem.add(opt.Shooting([0.0, 0.0], 1.0, 4, steps=["ctrl.damper.damping"]))
    nlp = problem.build()
    assert nlp.trajectory.stepped == {"ctrl.damper.damping": (3, 2, 2)}
    value = np.arange(12, dtype=float).reshape(3, 2, 2)  # three non-symmetric matrices
    x = problem.initial_guess({"steps": {"ctrl.damper.damping": value}})
    np.testing.assert_array_equal(nlp.unpack(x)["steps"]["ctrl.damper.damping"], value)
    result = problem.solve({}, warm_start=None)
    controller = vmc.VMCController(vmc.compile(system))
    result.steps["ctrl.damper.damping"] = value
    result.apply(controller, interval=2)
    np.testing.assert_array_equal(controller.live_params()["ctrl.damper.damping"], value[2])
    result.apply(controller)
    np.testing.assert_array_equal(controller.live_params()["ctrl.damper.damping"], value[0])
    with pytest.raises(ValueError, match="warm start steps"):
        problem.initial_guess({"steps": {"ctrl.damper.damping": value[:2]}})


def test_the_first_interval_starts_from_the_value_the_param_has_now():
    system, x = tanh_mass(stiffness=2.0, max_force=5.0, goal=1.0, bounds=(0.5, 50.0))
    k = "ctrl.spring.stiffness"
    problem = opt.Problem(system)
    problem.add(opt.Shooting([0.0], 1.0, 4, steps=[k]))
    problem.parameter(k)
    plan = problem.solve({k: 17.0})
    assert plan.references[k].item() == 17.0
    assert plan.steps[k].shape == (3,)
    assert (plan.steps[k] >= 0.5 - 1e-9).all() and (plan.steps[k] <= 50.0 + 1e-9).all()


def test_refusals():
    system, x = tanh_mass()
    k = "ctrl.spring.stiffness"
    with pytest.raises(ValueError, match="integrator"):
        opt.Shooting([0.0], 1.0, 3, integrator="cvodes")
    with pytest.raises(ValueError, match="nodes"):
        opt.Shooting([0.0], 1.0, 1)
    with pytest.raises(ValueError, match="horizon"):
        opt.Shooting([0.0], 0.0, 3)
    with pytest.raises(ValueError, match="substeps"):
        opt.Shooting([0.0], 1.0, 3, substeps=0)
    with pytest.raises(ValueError, match="scales"):
        opt.Shooting([0.0], 1.0, 3, scales=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="transition"):
        opt.Shooting([0.0], 1.0, 3, transition=-1.0)
    free = opt.Problem(system)
    free.add(opt.Shooting([0.0], 1.0, 3, steps=[k]))
    free.free(k)
    with pytest.raises(ValueError, match="free or steps"):
        free.build()
    unknown = opt.Problem(system)
    unknown.add(opt.Shooting([0.0], 1.0, 3, steps=["nothing"]))
    with pytest.raises(KeyError, match="no Param matches"):
        unknown.build()
    fixed = opt.Problem(system)
    fixed.add(opt.Shooting([0.0], 1.0, 3, steps=["robot.mass.inertance"]))
    with pytest.raises(ValueError, match="must be live"):
        fixed.build()
    wrong = opt.Problem(system)
    wrong.add(opt.Shooting([0.0, 0.0], 1.0, 3))
    with pytest.raises(ValueError, match="q0 and v0 need"):
        wrong.build()


def test_a_swap_from_the_controller_in_place_blends_as_the_collocation_does():
    def planned(block):
        system, x, _ = mass_spring(goal=1.0, stiffness=9.0)
        hold = vmc.Mechanism("hold")
        hold.add("spring", vmc.LinearSpring(x - 0.0, 4.0))
        hold.add("damper", vmc.LinearDamper(x, DAMPING))
        held = vmc.VirtualMechanismSystem(system.robot, hold)
        return plan_of(system, block(held))[1]

    shoot = planned(lambda h: opt.Shooting([0.0], 3.0, 13, initial=h, transition=2.0, substeps=20))
    colloc = planned(lambda h: opt.Collocation([0.0], 3.0, 241, initial=h, transition=2.0))
    assert shoot.blend[0] == 0.0 and shoot.blend[-1] == 1.0 and (np.diff(shoot.blend) >= 0).all()
    at_nodes = np.interp(shoot.t, colloc.t, colloc.q[:, 0])
    assert np.abs(shoot.q[:, 0] - at_nodes).max() < 0.02
    with pytest.raises(ValueError, match="same robot"):
        other, _, _ = mass_spring()
        plan_of(mass_spring()[0], opt.Shooting([0.0], 1.0, 3, initial=other))


def test_the_accelerations_of_the_nodes_are_those_of_the_robot_at_the_nodes():
    system, x, _ = mass_spring()
    _, plan = plan_of(system, opt.Shooting([0.0], 3.0, 61, substeps=2))
    # m a = -(friction + damping) v - stiffness (x - goal)
    expected = -(1.0 + DAMPING) * plan.v[:, 0] - 4.0 * (plan.q[:, 0] - 1.0)
    np.testing.assert_allclose(plan.a[:, 0], expected, atol=1e-9)
    assert plan.a.shape == plan.v.shape == plan.q.shape


def test_effort_counts_the_commands_at_the_nodes_and_the_last_one_uses_the_last_interval():
    system, x = tanh_mass(stiffness=2.0, max_force=5.0, goal=1.0, bounds=(0.5, 50.0))
    k = "ctrl.spring.stiffness"
    problem = opt.Problem(system)
    problem.add(opt.Shooting([0.0], 1.0, 5, steps=[k]))
    problem.add(opt.Effort(1.0))
    problem.parameter(k)
    plan = problem.solve({k: 2.0})
    last = plan.steps[k][-1]
    force = -5.0 * np.tanh(last * (plan.q[-1, 0] - 1.0) / 5.0) - DAMPING * plan.v[-1, 0]
    assert plan.u[-1, 0] == pytest.approx(force, rel=1e-6)
