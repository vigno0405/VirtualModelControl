"""Periodic orbits and a free horizon: a mass on a spring, with answers known in closed form."""

import dataclasses
from types import SimpleNamespace

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from small_plan import mass_spring
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.models import JointSpace

M, K, X = 1.0, 4.0, 0.3  # mass [kg], spring [N/m], amplitude [m]
PERIOD = 2.0 * np.pi * np.sqrt(M / K)
OMEGA = 2.0 * np.pi / PERIOD
HS, TRAPEZOID = "hermite-simpson", "trapezoid"


def oscillator():
    """A mass on a spring with no controller force, and its coordinate."""
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, M))
    robot.add("spring", vmc.LinearSpring(x, K))
    return vmc.VirtualMechanismSystem(robot, vmc.Mechanism("ctrl")), x


def orbit(horizon, nodes):
    """A guess of the orbit X cos(ωt), for a solver that starts at rest otherwise."""
    t = np.linspace(0.0, horizon, nodes)
    w = 2.0 * np.pi / horizon
    return {
        "q": (X * np.cos(w * t))[:, None],
        "v": (-X * w * np.sin(w * t))[:, None],
        "a": (-X * w**2 * np.cos(w * t))[:, None],
        "horizon": horizon,
    }


def swing(nodes, start, free_time=None, scheme=HS, guess=False, **kwargs):
    """The oscillator's orbit from x(0) = X: its horizon starts at ``start`` periods.

    The amplitude is pinned at the first node and a small cost on x keeps the smallest orbit that
    reaches it, so the orbit is the one with x(0) = X and v(0) = 0. At a fixed horizon the pin is
    only a lower bound: with an equality there are more equations than unknowns, which IPOPT
    does not accept (the cost holds it at the bound).
    """
    system, x = oscillator()
    problem = opt.Problem(system)
    problem.add(
        opt.Collocation(
            [X], start * PERIOD, nodes, periodic=True, free_time=free_time, scheme=scheme, **kwargs
        )
    )
    problem.add(opt.Bound(x, X, X if free_time else None, t_from=0.0, t_to=0.0, name="top"))
    problem.add(opt.Cost(x, name="small"))
    return problem.solve(warm_start=orbit(start * PERIOD, nodes) if guess else None), problem


def trapezoid_period(intervals):
    """The period of the trapezoid rule on the oscillator: it turns θ = 2 atan(ωh/2) per step."""
    return 2.0 * intervals * np.tan(np.pi / intervals) / OMEGA


class Duration(opt.Term):
    """What a cost on the horizon itself looks like: weight × the time of the motion."""

    name = "duration"

    def __init__(self, weight):
        self.weight = weight

    def cost(self, trajectory):
        return self.weight * trajectory.horizon


# (a) a free oscillation: the period is an unknown, and it is the natural one


@pytest.mark.parametrize("start", [0.8, 1.25])
def test_the_period_of_a_free_oscillation_is_found_with_hermite_simpson(start):
    r, _ = swing(40, start, free_time=(0.6 * PERIOD, 1.5 * PERIOD))
    assert r.converged and r.violation < 1e-6
    assert r.horizon == pytest.approx(PERIOD, rel=1e-3)
    assert r.t[-1] == pytest.approx(r.horizon, rel=1e-12) and r.t[0] == 0.0
    assert r.q[0, 0] == pytest.approx(X, abs=1e-9)  # the pinned start
    np.testing.assert_allclose(r.q[-1], r.q[0], atol=1e-9)  # the orbit repeats
    np.testing.assert_allclose(r.v[-1], r.v[0], atol=1e-9)
    energy = 0.5 * M * r.v[:, 0] ** 2 + 0.5 * K * r.q[:, 0] ** 2
    assert np.ptp(energy) < 1e-4 * energy.max()  # nothing leaves or enters
    assert np.abs(r.q[:, 0]).max() > 0.9 * X


def test_the_first_node_is_free_and_the_given_start_is_only_the_guess():
    # Pinned at 0.2 where the guess says 0.3: the guess is not the start.
    system, x = oscillator()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([X], PERIOD, 31, periodic=True, free_time=(2.0, 4.5), scheme=HS))
    problem.add(opt.Bound(x, 0.2, 0.2, t_from=0.0, t_to=0.0, name="top"))
    problem.add(opt.Cost(x, name="small"))
    r = problem.solve(warm_start=orbit(PERIOD, 31))
    assert r.converged
    assert r.q[0, 0] == pytest.approx(0.2, abs=1e-6)
    assert r.horizon == pytest.approx(PERIOD, rel=1e-3)


def test_the_trapezoid_rule_finds_its_own_discrete_period():
    # 39 intervals: the orbit the rule has is a rotation of 2 atan(ωh/2) a step, and the horizon
    # that closes it is 2 N tan(π/N) / ω, a little above the true period.
    r, _ = swing(40, 0.9, free_time=(0.6 * PERIOD, 1.5 * PERIOD), scheme=TRAPEZOID, guess=True)
    assert r.converged
    assert r.horizon == pytest.approx(trapezoid_period(39), rel=1e-6)
    assert r.horizon > PERIOD * (1.0 + 1e-3)


@pytest.mark.parametrize(("price", "edge"), [(1.0, 0), (-1.0, 1)])
def test_the_horizon_stays_within_its_bounds(price, edge):
    # A price on the time of the motion (negative: a reward) pushes the horizon to one end.
    bounds = (1.5, 4.5)
    system, _, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 3.0, 21, free_time=bounds))
    problem.add(Duration(price))
    r = problem.solve()
    assert r.converged and r.horizon == pytest.approx(bounds[edge], rel=1e-6)
    assert r.t[-1] == pytest.approx(bounds[edge], rel=1e-6)


# (b) a fixed horizon that is the period


def test_a_fixed_horizon_that_is_the_period_gives_back_the_oscillation():
    r, problem = swing(41, 1.0, guess=True)
    assert r.converged and r.violation < 1e-5
    assert r.horizon == PERIOD and r.t[-1] == pytest.approx(PERIOD, abs=1e-12)
    np.testing.assert_allclose(r.q[:, 0], X * np.cos(OMEGA * r.t), atol=1e-3 * X)
    np.testing.assert_allclose(r.q[-1], r.q[0], atol=1e-6)
    nlp = problem.build()
    assert "periodic" in nlp.constraints and "start" not in nlp.constraints
    assert "horizon" not in nlp.variables.slices  # nothing free in time


def test_a_fixed_horizon_that_is_not_the_period_has_no_orbit_of_that_amplitude():
    # Only at the period does the spring bring the mass back with the pinned start.
    r, _ = swing(41, 0.8, guess=True)
    assert not r.converged and r.violation > 0.1 * X


def test_the_trapezoid_orbit_closes_at_its_discrete_period_to_solver_accuracy():
    system, x = oscillator()
    problem = opt.Problem(system)
    nodes = 41
    problem.add(opt.Collocation([X], trapezoid_period(40), nodes, periodic=True, scheme=TRAPEZOID))
    problem.add(opt.Bound(x, X, None, t_from=0.0, t_to=0.0, name="top"))
    problem.add(opt.Cost(x, name="small"))
    r = problem.solve(warm_start=orbit(trapezoid_period(40), nodes))
    assert r.converged and r.violation < 1e-5
    energy = 0.5 * M * r.v[:, 0] ** 2 + 0.5 * K * r.q[:, 0] ** 2
    assert np.ptp(energy) < 1e-6 * energy.max()  # the rule keeps ½v² + ½kx² exactly


# (c) a driven oscillator: the periodic solution is the steady state


FRICTION, PULL, GOAL = 0.5, 10.0, 0.2  # [N s/m], [N/m], [m]


def driven(period):
    """The oscillator with friction, pulled by a spring to a goal that swings with ``period``."""
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, M))
    robot.add("spring", vmc.LinearSpring(x, K))
    robot.add("friction", vmc.LinearDamper(x, FRICTION))
    param = vmc.Param("period", period, unit="s", bounds=(1.0, 6.0), scope="episode")

    def goal(t, period):
        return GOAL * ca.cos(2.0 * np.pi * t / period)

    ref = vmc.Custom(goal, [vmc.Time()], dim=1, unit="m", params={"period": param})
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("pull", vmc.LinearSpring(x - ref, PULL))
    return vmc.VirtualMechanismSystem(robot, ctrl), x


def steady_state(t, period):
    """x(t) of the damped oscillator under the goal, long after the start (phasors)."""
    w = 2.0 * np.pi / period
    response = PULL * GOAL / (K + PULL - M * w**2 + 1j * FRICTION * w)
    return np.real(response * np.exp(1j * w * t))


@pytest.mark.parametrize("scheme", [HS, TRAPEZOID])
def test_the_periodic_solution_of_a_driven_oscillator_is_its_steady_state(scheme):
    period = 2.5
    errors = []
    for nodes in (21, 41):
        system, _ = driven(period)
        problem = opt.Problem(system)
        problem.add(opt.Collocation([0.0], period, nodes, periodic=True, scheme=scheme))
        r = problem.solve()
        assert r.converged and r.violation < 1e-8
        errors.append(np.abs(r.q[:, 0] - steady_state(r.t, period)).max())
    swing = np.abs(steady_state(np.linspace(0.0, period, 400), period)).max()
    assert swing > 0.2  # a real motion, not a rest
    if scheme == HS:
        assert errors[1] < 1e-5 * swing and errors[0] / errors[1] == pytest.approx(16.0, rel=0.3)
    else:
        assert errors[1] < 5e-3 * swing and errors[0] / errors[1] == pytest.approx(4.0, rel=0.1)


def test_the_periodic_orbit_does_not_depend_on_the_starting_guess():
    period = 2.5
    system, _ = driven(period)
    found = []
    for q0, v0 in (([0.0], [0.0]), ([0.7], [-0.4])):
        problem = opt.Problem(system)
        problem.add(opt.Collocation(q0, period, 41, v0=v0, periodic=True, scheme=HS))
        found.append(problem.solve())
    np.testing.assert_allclose(found[0].q, found[1].q, atol=1e-6)
    assert found[1].q[0, 0] != pytest.approx(0.7, abs=0.05)  # the first node is not the guess


# (d) the free period is optimized


def pulled(period0):
    """The oscillator with friction, pulled to cos and sin parts of a goal of a free period."""
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, M))
    robot.add("spring", vmc.LinearSpring(x, K))
    robot.add("friction", vmc.LinearDamper(x, 0.8))
    period = vmc.Param("period", period0, unit="s", bounds=(1.5, 6.0), scope="episode")
    cos = vmc.Param("cos", 0.0, unit="m", scope="episode")
    sin = vmc.Param("sin", 0.0, unit="m", scope="episode")

    def goal(t, period, cos, sin):
        phase = 2.0 * np.pi * t / period
        return cos * ca.cos(phase) + sin * ca.sin(phase)

    params = {"period": period, "cos": cos, "sin": sin}
    ref = vmc.Custom(goal, [vmc.Time()], dim=1, unit="m", params=params)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("pull", vmc.LinearSpring(x - ref, 10.0))
    return vmc.VirtualMechanismSystem(robot, ctrl), x


def least_effort_period(friction=0.8):
    """The period that moves the mass through the amplitude X with the least ∫u² over a period.

    The force is m x'' + c x' + k x with x = X cos(ωt): its mean square is
    X² ((k - m ω²)² + c² ω²) / 2, times the period T = 2π/ω. With s = ω², the derivative in T
    vanishes where 3 m² s² + (c² - 2 k m) s - k² = 0.
    """
    b = friction**2 - 2.0 * K * M
    s = (-b + np.sqrt(b**2 + 12.0 * M**2 * K**2)) / (6.0 * M**2)
    return 2.0 * np.pi / np.sqrt(s)


def effort(period, friction=0.8):
    w = 2.0 * np.pi / period
    return period / 2.0 * X**2 * ((K - M * w**2) ** 2 + (friction * w) ** 2)


def swing_with_period(start, bounds=(1.5, 6.0), scheme=HS, nodes=41):
    """The pulled oscillator's orbit through x(0) = X, with the period free within ``bounds``."""
    system, x = pulled(start)
    problem = opt.Problem(system)
    problem.add(opt.Collocation([X], start, nodes, periodic=True, free_time=bounds, scheme=scheme))
    problem.free("ctrl.pull.period", "ctrl.pull.cos", "ctrl.pull.sin")
    problem.add(opt.Period("ctrl.pull.period"))
    problem.add(opt.Bound(x, X, X, t_from=0.0, t_to=0.0, name="top"))
    problem.add(opt.Effort(1.0))
    return problem.solve(), problem


@pytest.mark.parametrize("start", [2.5, 4.5])
def test_the_least_effort_period_of_a_pulled_oscillator_is_found(start):
    best = least_effort_period()
    assert abs(best / PERIOD - 1.0) > 0.01  # not simply the period of the spring alone
    r, _ = swing_with_period(start)
    assert r.converged
    assert r.horizon == pytest.approx(best, rel=1e-3)
    assert r.params["ctrl.pull.period"].item() == pytest.approx(r.horizon, rel=1e-9)
    assert r.cost == pytest.approx(effort(best), rel=1e-3)
    assert abs(r.v[0, 0]) < 5e-3  # the smallest swing that reaches X starts at its turning point
    assert effort(best) < min(effort(0.95 * best), effort(1.05 * best))  # a true minimum


def test_the_period_bound_that_cuts_the_optimum_off_is_the_answer():
    best = least_effort_period()
    for bounds, expected in (((1.5, 0.9 * best), 0.9 * best), ((1.1 * best, 6.0), 1.1 * best)):
        r, _ = swing_with_period(0.5 * sum(bounds), bounds=bounds)
        assert r.converged
        assert r.horizon == pytest.approx(expected, rel=1e-6)
        assert r.params["ctrl.pull.period"].item() == pytest.approx(expected, rel=1e-6)


def test_a_cost_on_the_horizon_trades_effort_against_time():
    # The controller is a spring k_c to 0 (free), the mass and its own spring as before. With
    # x(0) = X and v(0) = 0 the orbit has the period of k + k_c, and the effort is
    # ∫ (k_c x)² dt = k_c² X² T / 2. With a price w on each second the total is
    # (k_c² X² / 2 + w) T, whose minimum has 3 X² k_c² + 4 X² k k_c - 2 w = 0.
    price = 0.5
    best = (-4.0 * X**2 * K + np.sqrt(16.0 * X**4 * K**2 + 24.0 * X**2 * price)) / (6.0 * X**2)
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, M))
    robot.add("spring", vmc.LinearSpring(x, K))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("pull", vmc.LinearSpring(x, 5.0))
    problem = opt.Problem(vmc.VirtualMechanismSystem(robot, ctrl))
    problem.add(opt.Collocation([X], 1.5, 41, periodic=True, free_time=(0.5, 4.0), scheme=HS))
    problem.free("ctrl.pull.stiffness")
    problem.add(opt.Bound(x, X, X, t_from=0.0, t_to=0.0, name="top"))
    problem.add(opt.Effort(1.0))
    problem.add(Duration(price))
    r = problem.solve(warm_start=orbit(1.5, 41))
    assert r.converged
    assert r.params["ctrl.pull.stiffness"].item() == pytest.approx(best, rel=1e-3)
    assert r.horizon == pytest.approx(2.0 * np.pi * np.sqrt(M / (K + best)), rel=1e-3)
    assert r.costs["duration"] == pytest.approx(price * r.horizon, rel=1e-12)


# the terms and the coordinates see the free horizon


def free_nlp(nodes=11, start=3.0):
    system, x, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], start, nodes, free_time=(1.0, 5.0)))
    return problem, x


def test_the_node_times_and_spacing_follow_the_free_horizon_and_t_stays_the_starting_one():
    problem, _ = free_nlp()
    nlp = problem.build()
    trajectory = nlp.trajectory
    np.testing.assert_allclose(trajectory.t, np.linspace(0.0, 3.0, 11))  # the nominal times
    horizon = nlp.variables.slices["horizon"]
    x = nlp.x0.copy()
    x[horizon] = 4.0 / nlp.variables.scales["horizon"]  # the solver's value for T = 4 s
    at = lambda expr: float(ca.Function("f", [nlp.x, nlp.p], [expr])(x, []))  # noqa: E731
    assert at(trajectory.dt) == pytest.approx(0.4)
    assert at(trajectory.horizon) == pytest.approx(4.0)
    assert [at(t) for t in trajectory.times] == pytest.approx(list(np.linspace(0.0, 4.0, 11)))
    assert trajectory.times[0] == 0.0


def test_a_fixed_trajectory_has_numbers_for_its_times_and_horizon():
    system, _, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 3.0, 11))
    trajectory = problem.build().trajectory
    assert trajectory.times == pytest.approx(list(trajectory.t))
    assert all(isinstance(t, float) for t in trajectory.times)
    assert trajectory.horizon == pytest.approx(3.0)
    assert isinstance(trajectory.dt, float)


def test_effort_and_cost_integrate_with_the_horizon_that_was_found():
    system, x, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 3.0, 31, free_time=(1.5, 6.0)))
    problem.free("ctrl.spring.stiffness")
    problem.add(opt.Effort(0.7))
    problem.add(opt.Cost(x - 1.0, 2.0, t_from=1.0, name="reach"))
    r = problem.solve()
    assert r.converged and abs(r.horizon - 3.0) > 0.1  # it moved: the check is not trivial
    dt = r.horizon / 30
    u2 = r.u[:, 0] ** 2
    expected = 0.7 * dt * (0.5 * u2[0] + u2[1:-1].sum() + 0.5 * u2[-1])
    assert r.costs["effort"] == pytest.approx(expected, rel=1e-12)
    # The window is chosen on the starting horizon (t >= 1 s of 3 s: nodes 10 to 30).
    y2 = (r.q[10:, 0] - 1.0) ** 2
    assert r.costs["reach"] == pytest.approx(2.0 * dt * (y2[:-1].sum() + 0.5 * y2[-1]), rel=1e-12)


def test_a_coordinate_of_time_is_evaluated_at_the_free_times():
    system, x, _ = mass_spring()
    clock = vmc.Custom(lambda t: t - 1.0, [vmc.Time()], dim=1, unit="s")
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 3.0, 31, free_time=(2.0, 4.0)))
    problem.add(opt.Cost(clock, 1.0, name="clock"))
    problem.add(opt.Cost(x, 1.0, name="stay"))
    r = problem.solve()
    assert r.converged and r.horizon < 2.5  # the clock cost wants the shortest horizon
    dt = r.t[1] - r.t[0]
    y2 = (r.t - 1.0) ** 2
    assert r.costs["clock"] == pytest.approx(dt * (y2[:-1].sum() + 0.5 * y2[-1]), rel=1e-10)


def test_the_controller_sees_the_free_times():
    # A goal that moves with time: the force at node k is that of the time k T / (N - 1).
    robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
    x = robot.joint(0)
    robot.add("mass", vmc.Inertance(x, M))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("pull", vmc.LinearSpring(x - vmc.Custom(lambda t: 0.1 * t, [vmc.Time()], dim=1), 3.0))
    problem = opt.Problem(vmc.VirtualMechanismSystem(robot, ctrl))
    problem.add(opt.Collocation([0.0], 2.0, 21, free_time=(1.0, 3.0), scheme=HS))
    problem.add(opt.Cost(x - 0.5, name="reach"))
    r = problem.solve()
    assert r.converged and r.horizon != pytest.approx(2.0, abs=1e-3)
    np.testing.assert_allclose(r.u[:, 0], -3.0 * (r.q[:, 0] - 0.1 * r.t), atol=1e-7)


@pytest.mark.parametrize("scheme", [HS, TRAPEZOID])
def test_a_free_horizon_gives_the_plan_of_a_fixed_one_at_the_horizon_it_found(scheme):
    def plan(horizon, free_time):
        robot = vmc.Mechanism("robot", model=JointSpace(1, unit="m"))
        x = robot.joint(0)
        robot.add("mass", vmc.Inertance(x, M))
        robot.add("friction", vmc.LinearDamper(x, 0.5))
        ctrl = vmc.Mechanism("ctrl")
        goal = vmc.Custom(lambda t: 0.3 * t, [vmc.Time()], dim=1, unit="m")
        ctrl.add("pull", vmc.LinearSpring(x - goal, 3.0))
        problem = opt.Problem(vmc.VirtualMechanismSystem(robot, ctrl))
        problem.add(opt.Collocation([0.0], horizon, 21, free_time=free_time, scheme=scheme))
        # Be at 0.5 at the last node (a window on the starting horizon's last node), soon.
        problem.add(opt.Cost(x - 0.5, 1000.0, t_from=horizon, t_to=horizon, name="arrive"))
        problem.add(Duration(0.05))
        return problem.solve()

    free = plan(2.0, (0.5, 8.0))
    assert free.converged and 0.6 < free.horizon < 7.9 and free.horizon != pytest.approx(2.0)
    fixed = plan(free.horizon, None)
    assert fixed.converged and fixed.horizon == free.horizon
    assert fixed.horizon not in (0.5, 8.0)
    np.testing.assert_allclose(fixed.q, free.q, atol=1e-5)
    np.testing.assert_allclose(fixed.u, free.u, atol=1e-4)
    assert fixed.cost == pytest.approx(free.cost, rel=1e-6)


# the result, the warm start and the arguments


def test_a_result_reports_the_horizon_that_was_planned():
    system, _, _ = mass_spring()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 3.0, 11))
    fixed = problem.solve()
    assert fixed.horizon == 3.0 and fixed.t[-1] == pytest.approx(3.0, abs=1e-12)
    rest = opt.Problem(system)
    rest.add(opt.Equilibrium([0.0]))
    there = rest.solve()
    assert there.horizon == 0.0 and there.t.tolist() == [0.0]


def test_a_warm_start_carries_the_horizon_of_a_free_problem():
    problem, _ = free_nlp()
    nlp = problem.build()
    horizon = lambda x: nlp.variables.value(x, "horizon").item()  # noqa: E731
    assert horizon(problem.initial_guess()) == pytest.approx(3.0)
    assert horizon(problem.initial_guess({"horizon": 4.5})) == pytest.approx(4.5)
    later = dataclasses.replace(problem.solve(), horizon=4.5)
    assert horizon(problem.initial_guess(later)) == pytest.approx(4.5)
    where = nlp.variables.slices["horizon"]
    assert nlp.lbx[where] == pytest.approx(1.0 / 3.0)  # the bounds in the solver's units
    assert nlp.ubx[where] == pytest.approx(5.0 / 3.0)
    system, _, _ = mass_spring()
    fixed = opt.Problem(system)
    fixed.add(opt.Collocation([0.0], 3.0, 11))
    fixed.initial_guess({"horizon": 4.5})  # a fixed horizon has nothing to set: not an error


def test_the_start_of_a_periodic_problem_is_not_fixed_and_a_bound_reaches_the_first_node():
    system, x = oscillator()
    for periodic, error in ((True, None), (False, "no node lies in the window")):
        problem = opt.Problem(system)
        problem.add(opt.Collocation([X], PERIOD, 11, periodic=periodic))
        problem.add(opt.Bound(x, X, None, t_from=0.0, t_to=0.0))
        if error is None:
            assert problem.build().trajectory.fixed_start is False
        else:
            with pytest.raises(ValueError, match=error):
                problem.build()


def test_the_arguments_of_a_free_horizon_and_a_periodic_motion_are_checked():
    with pytest.raises(ValueError, match="free_time is"):
        opt.Collocation([0.0], 3.0, 10, free_time=3.0)
    with pytest.raises(ValueError, match="free_time is"):
        opt.Collocation([0.0], 3.0, 10, free_time=(1.0, 2.0, 4.0))
    with pytest.raises(ValueError, match="free_time is"):
        opt.Collocation([0.0], 3.0, 10, free_time=(4.0, 2.0))
    with pytest.raises(ValueError, match="free_time is"):
        opt.Collocation([0.0], 3.0, 10, free_time=(3.0, 3.0))
    with pytest.raises(ValueError, match="free_time is"):
        opt.Collocation([0.0], 0.5, 10, free_time=(-1.0, 4.0))
    with pytest.raises(ValueError, match="starting value"):
        opt.Collocation([0.0], 3.0, 10, free_time=(1.0, 2.0))
    with pytest.raises(ValueError, match="starting value"):
        opt.Collocation([0.0], 3.0, 10, free_time=(3.5, 6.0))
    opt.Collocation([0.0], 2.0, 10, free_time=(2.0, 3.0))  # the ends are inside
    opt.Collocation([0.0], 3.0, 10, free_time=(2.0, 3.0))


def test_a_free_horizon_does_not_go_with_a_controller_in_place():
    system, x, _ = mass_spring()
    hold = vmc.Mechanism("hold")
    hold.add("spring", vmc.LinearSpring(x - 0.0, 4.0))
    held = vmc.VirtualMechanismSystem(system.robot, hold)
    with pytest.raises(ValueError, match="free_time cannot go with initial or transition"):
        opt.Collocation([0.0], 3.0, 10, free_time=(1.0, 4.0), initial=held, transition=1.0)
    with pytest.raises(ValueError, match="free_time cannot go with initial or transition"):
        opt.Collocation([0.0], 3.0, 10, free_time=(1.0, 4.0), initial=held)
    with pytest.raises(ValueError, match="free_time cannot go with initial or transition"):
        opt.Collocation([0.0], 3.0, 10, free_time=(1.0, 4.0), transition=1.0)
    with pytest.raises(ValueError, match="periodic motion repeats"):
        opt.Collocation([0.0], 3.0, 10, periodic=True, initial=held)


def test_a_periodic_motion_needs_a_flat_space_as_every_collocation_does():
    space = SimpleNamespace(space=vmc.SO2())
    builder = SimpleNamespace(system=SimpleNamespace(robot=SimpleNamespace(model=space)))
    with pytest.raises(NotImplementedError, match="Euclidean"):
        opt.Collocation([0.0], 3.0, 10, periodic=True, free_time=(1.0, 4.0)).build(builder)


# the period term


def test_the_period_term_names_one_scalar_param():
    system, _ = driven(2.5)
    for name, match in (
        ("nothing.here", "matches 0 Params"),
        ("ctrl.pull.*", "matches 2 Params"),
    ):
        problem = opt.Problem(system)
        problem.add(opt.Collocation([0.0], 2.5, 11, periodic=True))
        problem.add(opt.Period(name))
        with pytest.raises(ValueError, match=match):
            problem.build()
    vector = vmc.Mechanism("robot", model=JointSpace(2, unit="m"))
    q = vector.joint(slice(0, 2))
    vector.add("mass", vmc.Inertance(q, M))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("pull", vmc.LinearSpring(q - vmc.Ref("goal", 2, [0.0, 0.0]), 1.0))
    problem = opt.Problem(vmc.VirtualMechanismSystem(vector, ctrl))
    problem.add(opt.Collocation([0.0, 0.0], 2.5, 11, periodic=True))
    problem.add(opt.Period("ctrl.pull.goal"))
    with pytest.raises(ValueError, match="one number"):
        problem.build()
    with pytest.raises(ValueError, match="before the term"):
        opt.Period("x").build(SimpleNamespace(trajectory=None))


def test_the_period_term_ties_the_horizon_to_a_param_of_the_same_value():
    system, _ = driven(2.5)
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 2.5, 11, periodic=True, free_time=(1.5, 3.5)))
    problem.parameter("ctrl.pull.period")
    problem.add(opt.Period("ctrl.pull.period"))
    nlp = problem.build()
    assert "period" in nlp.constraints
    for period in (2.0, 3.0):
        r = problem.solve({"ctrl.pull.period": period})
        assert r.converged and r.horizon == pytest.approx(period, rel=1e-9)
