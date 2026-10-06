"""Dither seeking: the minimum of a measured cost found by two Params dithered at two tones."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import DitherSeeking
from virtualmodelcontrol.control import Tank

DT = 0.01
NAMES = ["ctrl.a.stiffness", "ctrl.b.stiffness"]
OPTIMUM = np.array([3.0, 1.5])
CURVATURE = np.array([2.0, 0.5])


def controller(start=(1.0, 2.5)):
    robot = vmc.Mechanism("robot", model=vmc.models.JointSpace(2))
    ctrl = vmc.Mechanism("ctrl")
    for name, k, i in (("a", start[0], 0), ("b", start[1], 1)):
        goal = vmc.Ref(f"goal_{name}", 1, value=[1.0])  # a deflection, so a stiffness stores energy
        ctrl.add(name, vmc.LinearSpring(robot.joint(i) - goal, k))
    return vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))


def cost(values, optimum=OPTIMUM):
    """A bowl with its minimum at ``optimum``: what a run would measure."""
    return float(np.sum(CURVATURE * (values - optimum) ** 2))


def run(c, law, seconds=40.0, noise=0.0, seed=0, optimum=OPTIMUM):
    rng = np.random.default_rng(seed)
    jumps, history = [], []
    rest = vmc.Signals(0.0, motor_position=np.zeros(2), motor_velocity=np.zeros(2))
    c.reset(0.0, rest)
    for k in range(int(seconds / DT)):
        t = k * DT
        c.step(t, vmc.Signals(t, motor_position=np.zeros(2), motor_velocity=np.zeros(2)))
        live = c.live_params()
        values = np.array([float(np.ravel(live[n])[0]) for n in NAMES])
        jumps.append(law.step(c, cost(values, optimum) + rng.normal(0, noise), t))
        history.append(law.estimate.copy())
    return np.array(history), np.array(jumps)


def law_for(c, **kwargs):
    options = dict(
        amplitude=[0.3, 0.3],
        frequency=[2 * np.pi * 2 / 4, 2 * np.pi * 3 / 4],
        gain=[0.05, 0.2],
        window=4.0,
    )
    return DitherSeeking(c, NAMES, **{**options, **kwargs})


def test_two_params_find_the_minimum_of_a_bowl_from_its_cost_alone():
    c = controller()
    history, _ = run(c, law_for(c))
    np.testing.assert_allclose(history[-1], OPTIMUM, atol=0.15)
    assert abs(history[0] - OPTIMUM).max() > 1.0  # it started far


def test_the_dither_is_what_is_set_around_the_estimate_at_each_tone():
    c = controller()
    law = law_for(c)
    t = 0.37
    law.step(c, 0.0, t)
    live = c.live_params()
    got = np.array([float(np.ravel(live[n])[0]) for n in NAMES])
    want = law.estimate + np.array([0.3, 0.3]) * np.sin(law.frequency * t)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_the_estimate_moves_at_most_max_rate_and_stays_within_the_params_bounds():
    c = controller()
    law = law_for(c, max_rate=0.2)
    history, _ = run(c, law, seconds=15.0)
    assert np.abs(np.diff(history, axis=0)).max() <= 0.2 * DT * (1 + 1e-9)
    c = controller()
    law = law_for(c)
    history, _ = run(c, law, optimum=np.array([-5.0, 1.5]))  # outside the bound at zero
    assert history[:, 0].min() >= 0.0 and history[-1, 0] < 0.1  # it rests at the bound


def test_a_noisy_cost_still_leads_to_the_minimum_on_average():
    c = controller()
    history, _ = run(c, law_for(c), seconds=80.0, noise=0.3)
    np.testing.assert_allclose(history[len(history) // 2 :].mean(axis=0), OPTIMUM, atol=0.3)


def test_a_tank_pays_for_the_changes_and_never_goes_below_empty():
    c = controller()
    tank = Tank(c, level=0.0, capacity=0.005)
    _, jumps = run(tank, law_for(tank), seconds=20.0)
    assert tank.level >= -1e-12  # nothing is given that the tank does not hold
    assert jumps.max() <= 0.005 + 1e-9  # no jump above what it ever held
    free = controller()
    _, free_jumps = run(free, law_for(free), seconds=20.0)
    assert free_jumps.max() > 0.005  # without it the dither injects more energy than that


def test_the_params_must_be_live_and_scalar():
    c = controller()
    with pytest.raises(ValueError, match="no live Param"):
        DitherSeeking(c, "ctrl.*.nothing", 0.1, 5.0, 1.0, 2.0)
    robot = vmc.Mechanism("robot", model=vmc.models.JointSpace(2))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("pair", vmc.LinearSpring(robot.joint(slice(0, 2)) - vmc.Ref("goal", 2), [1.0, 2.0]))
    both = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    with pytest.raises(ValueError, match="scalar"):
        DitherSeeking(both, "ctrl.pair.stiffness", 0.1, 5.0, 1.0, 2.0)


def test_the_slope_is_the_costs_own_slope_and_a_constant_in_the_cost_does_not_matter():
    for offset in (0.0, 100.0):
        c = controller()
        law = law_for(c, gain=0.0)  # the estimate stays where it starts: (1, 2.5)
        c.reset(0.0, vmc.Signals(0.0, motor_position=np.zeros(2), motor_velocity=np.zeros(2)))
        for k in range(int(12.0 / DT)):
            t = k * DT
            live = c.live_params()
            values = np.array([float(np.ravel(live[n])[0]) for n in NAMES])
            law.step(c, cost(values) + offset, t)
        true = 2 * CURVATURE * (law.estimate - OPTIMUM)  # (-8, 2)
        np.testing.assert_allclose(law.slope, true, rtol=0.08, err_msg=f"offset {offset}")


def test_nothing_moves_until_the_window_has_filled_and_the_dither_stays_in_bounds():
    c = controller()
    law = law_for(c)
    c.reset(0.0, vmc.Signals(0.0, motor_position=np.zeros(2), motor_velocity=np.zeros(2)))
    applied = []
    for k in range(int(10.0 / DT)):
        t = k * DT
        live = c.live_params()
        applied.append([float(np.ravel(live[n])[0]) for n in NAMES])
        law.step(c, cost(np.array(applied[-1]), np.array([-5.0, 1.5])), t)
        if t < 0.85 * law.window:
            np.testing.assert_array_equal(law.estimate, [1.0, 2.5])
    assert np.min(applied) >= 0.0  # a stiffness is never set below its bound
