"""MPC: a mass regulated to a goal under a force cap, planned again at every step."""

import threading

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.models import JointSpace
from virtualmodelcontrol.optimization import mpc as mpc_module

K, GOAL = "ctrl.spring.stiffness", "ctrl.spring.goal"
H, PERIOD, DT = 0.02, 5, 0.1  # a control step [s], control steps per plan step, an interval [s]


def mass(goal=0.0):
    """A mass under a spring of force cap 5 N, whose stiffness and reference are bounded Params."""
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, 1.0))
    robot.add("friction", vmc.LinearDamper(x, 1.0))
    ref = vmc.Ref("goal", 1, vmc.Param("goal", [goal], bounds=(-1.0, 2.0), scope="stage"))
    k = vmc.Param("stiffness", 2.0, bounds=(0.5, 50.0), scope="stage")
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("spring", vmc.TanhSpring(x - ref, k, 5.0))
    damping = vmc.Param("damping", 1.0, bounds=(0.2, 4.0), scope="stage")
    ctrl.add("damper", vmc.LinearDamper(x, damping))
    return vmc.VirtualMechanismSystem(robot, ctrl), x


def program(level=None, budget=True, free=(), **kwargs):
    """A mass pulled to ``target`` by a spring (force cap 5 N) whose stiffness and reference are
    planned over 1 s; ``level`` is the tank that pays for their changes."""
    system, x = mass()
    problem = opt.Problem(system, solver="ipopt-exact")
    problem.add(opt.Shooting([0.0], 1.0, 11, steps=[K, GOAL], substeps=PERIOD))
    problem.add(opt.Effort(0.01))
    target = vmc.Ref("target", 1, [1.0])
    problem.add(opt.Cost(x - target, 1.0, name="reach"))
    if level is not None and budget:
        problem.add(opt.TankBudget(level))
    problem.parameter("reach.target")
    problem.free(*free)
    return system, problem, opt.MPC(problem, **kwargs)


def loop(system, mpc, seconds, level=None, cold=False, latency=0.0, references=None):
    """The mass under the controller, the plan applied every ``PERIOD`` control steps."""
    plant = vmc.sim.ModelPlant(system.robot, q0=[0.0], max_step=H)
    controller = vmc.VMCController(vmc.compile(system))
    ctrl = controller if level is None else vmc.control.Tank(controller, level=level)
    ctrl.reset(0.0, plant.read())
    log = []
    for i in range(round(seconds / H)):
        meas = plant.read()
        cmd = ctrl.step(plant.t, meas)
        if i % PERIOD == 0:
            if cold:
                mpc.reset()
            refs = references(plant.t) if references else None
            jump = mpc.step(ctrl, plant.q, plant.v, refs, latency=latency)
            log.append((plant.t, plant.q[0], jump, mpc.result.iterations, mpc.result.converged))
        plant.write(cmd)
        plant.advance(H)
    return ctrl, plant, np.array(log, dtype=float)


def test_the_mass_reaches_the_goal_under_the_force_cap():
    system, _problem, mpc = program()
    _, plant, log = loop(system, mpc, 4.0)
    assert abs(plant.q[0] - 1.0) < 5e-3 and abs(plant.v[0]) < 5e-2
    assert (log[:, 4] == 1.0).all()  # every plan converged
    assert log[:, 1].max() < 1.1  # it does not run far past the goal either
    assert mpc.intervals == 10 and mpc.dt == pytest.approx(DT) and mpc.names == [K, GOAL]


def test_the_program_is_built_once_and_the_references_are_read_at_every_step():
    system, problem, mpc = program()
    nlp = mpc.nlp
    ctrl, plant, log = loop(
        system, mpc, 6.0, references=lambda t: {"reach.target": [1.0 if t < 3.0 else -0.5]}
    )
    assert mpc.nlp is nlp is problem.build()
    assert mpc.result.references[K].item() == pytest.approx(ctrl.live_params()[K].item(), rel=0.5)
    assert abs(plant.q[0] - (-0.5)) < 1e-2  # it follows the new target
    assert abs(log[round(3.0 / (PERIOD * H)) - 1, 1] - 1.0) < 1e-2  # and had arrived at the first


def test_a_warm_start_from_the_shifted_plan_needs_fewer_iterations():
    system, _, warm = program()
    _, _, warm_log = loop(system, warm, 3.0)
    system, _, cold = program()
    _, _, cold_log = loop(system, cold, 3.0, cold=True)
    assert warm_log[:, 3].sum() < 0.8 * cold_log[:, 3].sum()
    assert abs(warm_log[-1, 1] - cold_log[-1, 1]) < 1e-2  # the same closed loop


def test_the_plan_moves_up_by_the_shift_with_its_last_values_held():
    for shift in (1, 3):
        system, _problem, mpc = program(shift=shift)
        _ctrl, _plant, _ = loop(system, mpc, 0.6)
        plan, warm = mpc.result, mpc._warm
        for j in range(11):
            np.testing.assert_array_equal(warm["q"][j], plan.q[min(j + shift, 10)])
            np.testing.assert_array_equal(warm["v"][j], plan.v[min(j + shift, 10)])
        for j in range(10):
            for name in (K, GOAL):
                np.testing.assert_array_equal(
                    warm["steps"][name][j], plan.steps[name][min(j + shift, 9)]
                )


@pytest.mark.parametrize(
    "latency, interval", [(0.0, 0), (0.05, 0), (0.1, 1), (0.35, 3), (0.55, 5), (7.0, 9)]
)
def test_a_late_plan_is_applied_as_of_the_time_it_was_late(latency, interval):
    system, _problem, mpc = program()
    controller = vmc.VMCController(vmc.compile(system))
    plant = vmc.sim.ModelPlant(system.robot, q0=[0.0])
    controller.reset(0.0, plant.read())
    controller.step(0.0, plant.read())
    mpc.step(controller, plant.q, plant.v, latency=latency)
    assert mpc.interval == interval
    live = controller.live_params()
    for name in (K, GOAL):
        np.testing.assert_allclose(live[name].ravel(), mpc.result.steps[name][interval].ravel())


def test_the_measured_solve_time_is_the_latency_by_default(monkeypatch):
    system, problem, mpc = program()
    solve = problem.solve

    def slow(*args, **kwargs):
        result = solve(*args, **kwargs)
        result.seconds = 0.31  # as if the solver had been slow
        return result

    monkeypatch.setattr(problem, "solve", slow)
    controller = vmc.VMCController(vmc.compile(system))
    plant = vmc.sim.ModelPlant(system.robot, q0=[0.0])
    controller.reset(0.0, plant.read())
    mpc.step(controller, plant.q, plant.v)
    assert mpc.interval == 3


def test_the_first_plan_starts_from_the_measured_state_and_the_next_from_the_shifted_plan(
    monkeypatch,
):
    system, problem, mpc = program(free=["ctrl.damper.damping"])  # one value for the whole horizon
    seen, solve = [], problem.solve

    def spy(references=None, warm_start=None, **kwargs):
        seen.append(warm_start)
        return solve(references, warm_start, **kwargs)

    monkeypatch.setattr(problem, "solve", spy)
    controller = vmc.VMCController(vmc.compile(system))
    plant = vmc.sim.ModelPlant(system.robot, q0=[0.4], v0=[0.3])
    controller.reset(0.0, plant.read())
    controller.step(0.0, plant.read())
    now = {name: np.array(value) for name, value in controller.live_params().items()}
    mpc.step(controller, plant.q, plant.v, latency=0.0)
    first, damping = seen[0], mpc.result.params["ctrl.damper.damping"]
    np.testing.assert_array_equal(first["q"], np.full((11, 1), 0.4))
    np.testing.assert_array_equal(first["v"], np.full((11, 1), 0.3))
    assert set(first["steps"]) == {K, GOAL}
    for name, rows in first["steps"].items():
        assert rows.shape == (10, *now[name].shape)
        np.testing.assert_array_equal(rows[3], now[name])  # held at what the controller has
    mpc.step(controller, plant.q, plant.v, latency=0.0)
    np.testing.assert_array_equal(seen[1]["params"]["ctrl.damper.damping"], damping)
    assert 0.2 <= damping.item() <= 4.0
    np.testing.assert_allclose(
        controller.live_params()["ctrl.damper.damping"], mpc.result.params["ctrl.damper.damping"]
    )


def test_the_energy_it_returns_is_the_jump_of_the_controllers_energy():
    system, _problem, mpc = program()
    controller = vmc.VMCController(vmc.compile(system))
    plant = vmc.sim.ModelPlant(system.robot, q0=[0.4], v0=[0.3])
    controller.reset(0.0, plant.read())
    controller.step(0.0, plant.read())
    before = controller.energy()
    jump = mpc.step(controller, plant.q, plant.v, latency=0.0)
    assert jump == pytest.approx(controller.energy() - before, abs=1e-12) and abs(jump) > 1e-3
    assert mpc.result.status == "Solve_Succeeded"


def tank_run(budget):
    """A tight tank (0.05 J): the fraction of each change that the tank paid, and the mass."""
    system, _problem, mpc = program(level=0.05, budget=budget)
    tank = vmc.control.Tank(vmc.VMCController(vmc.compile(system)), level=0.05)
    plant = vmc.sim.ModelPlant(system.robot, q0=[0.0], max_step=H)
    tank.reset(0.0, plant.read())
    fractions, levels = [], []
    for i in range(round(3.0 / H)):
        meas = plant.read()
        cmd = tank.step(plant.t, meas)
        if i % PERIOD == 0:
            mpc.step(tank, plant.q, plant.v, latency=0.0)
            fractions.append(tank.fraction)
        levels.append(tank.level)
        plant.write(cmd)
        plant.advance(H)
    return np.array(fractions), np.array(levels), plant.q[0]


def test_a_plan_with_the_budget_is_executed_whole_by_a_tight_tank_and_one_without_is_cut():
    fractions, levels, x = tank_run(budget=True)
    assert (fractions == 1.0).all() and levels.min() > -1e-9
    assert x > 0.15  # it moves on as far as the energy of the tank allows
    fractions, levels, _ = tank_run(budget=False)
    assert fractions.min() < 0.5 and levels.min() > -1e-9  # the tank keeps its budget by cutting


def test_the_real_time_iteration_takes_one_sqp_step_per_plan_and_still_reaches_the_goal():
    system, problem, mpc = program(rti=True)
    assert problem.solver == "rti"
    _, plant, log = loop(system, mpc, 5.0)
    assert (log[:, 3] == 1.0).all()
    assert abs(plant.q[0] - 1.0) < 3e-2


def test_refusals():
    system, _ = mass()
    plain = opt.Problem(system)
    plain.add(opt.Collocation([0.0], 1.0, 4))
    with pytest.raises(ValueError, match="Shooting"):
        opt.MPC(plain)
    with pytest.raises(ValueError, match="shift"):
        program(shift=0)
    system, _problem, mpc = program(level=1.0)
    with pytest.raises(AttributeError):  # a tank is needed to read the level from
        mpc.step(vmc.VMCController(vmc.compile(system)), [0.0], [0.0])


def ready(system):
    """A controller that has taken its first step, and its plant."""
    controller = vmc.VMCController(vmc.compile(system))
    plant = vmc.sim.ModelPlant(system.robot, q0=[0.0], max_step=H)
    controller.reset(0.0, plant.read())
    controller.step(0.0, plant.read())
    return controller, plant


def test_a_plan_made_in_a_thread_is_applied_when_it_is_ready_as_of_the_time_it_took(monkeypatch):
    system, problem, mpc = program()
    controller, plant = ready(system)
    gate, solve = threading.Event(), problem.solve

    def held(*args, **kwargs):
        gate.wait(5.0)
        return solve(*args, **kwargs)

    monkeypatch.setattr(problem, "solve", held)
    clock = [10.0]
    monkeypatch.setattr(mpc_module.time, "perf_counter", lambda: clock[0])
    assert mpc.poll(controller) is None  # nothing started
    assert mpc.start(controller, plant.q, plant.v) is True
    assert mpc.busy and mpc.start(controller, plant.q, plant.v) is False  # one plan at a time
    assert mpc.poll(controller) is None  # not ready: the controller is not touched
    assert controller.live_params()[K] == 2.0
    clock[0] = 10.25
    gate.set()
    mpc._thread.join()
    assert not mpc.busy
    jump = mpc.poll(controller)  # a quarter of a second after the start: the third interval
    assert isinstance(jump, float) and mpc.interval == 2
    np.testing.assert_allclose(controller.live_params()[K], mpc.result.steps[K][2])
    assert mpc.poll(controller) is None  # applied once
    assert mpc.start(controller, plant.q, plant.v) is True  # and the next can start
    mpc._thread.join()
    assert mpc.poll(controller, latency=0.0) is not None and mpc.interval == 0


def test_the_solver_is_created_with_the_mpc_not_in_the_thread_that_plans():
    system, problem, mpc = program()
    kept = problem._solver
    assert kept is not None  # CasADi is not safe to build a solver in a second thread
    controller, plant = ready(system)
    mpc.start(controller, plant.q, plant.v)
    mpc._thread.join()
    assert problem._solver is kept


def test_a_solve_that_fails_in_the_thread_fails_in_the_control_loop(monkeypatch):
    system, problem, mpc = program()
    controller, plant = ready(system)

    def broken(*args, **kwargs):
        raise RuntimeError("no plan")

    monkeypatch.setattr(problem, "solve", broken)
    mpc.start(controller, plant.q, plant.v)
    mpc._thread.join()
    with pytest.raises(RuntimeError, match="no plan"):
        mpc.poll(controller)
    assert mpc.poll(controller) is None and not mpc.busy


def test_the_loop_with_plans_made_in_threads_reaches_the_goal():
    system, _problem, mpc = program()
    plant = vmc.sim.ModelPlant(system.robot, q0=[0.0], max_step=H)
    controller = vmc.VMCController(vmc.compile(system))
    controller.reset(0.0, plant.read())
    for i in range(round(4.0 / H)):
        command = controller.step(plant.t, plant.read())
        if i % PERIOD == 0:
            mpc.start(controller, plant.q, plant.v)
            mpc._thread.join()  # the simulation has no clock: wait for the plan
            mpc.poll(controller, latency=0.0)
        plant.write(command)
        plant.advance(H)
    assert abs(plant.q[0] - 1.0) < 5e-3
