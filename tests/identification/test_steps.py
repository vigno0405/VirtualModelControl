from functools import cache

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.identification import Steps, fit_stiffness_damping, validate
from virtualmodelcontrol.robots import helyx

DT = 0.01


class Still:
    """A plant that does not move, with three motors."""

    t = 0.0

    def read(self):
        return vmc.Signals(self.t, motor_position=np.zeros(3), motor_velocity=np.zeros(3))

    def write(self, cmd):
        pass

    def advance(self, dt):
        self.t += dt


def run(steps, T=None, plant=None):
    plant = Still() if plant is None else plant
    T = steps.duration if T is None else T
    return vmc.sim.run(plant, steps, vmc.sim.SimClock(DT), T=T).arrays()


def torque(rows, t):
    return rows["motor_torque"][round(t / DT)]


def test_every_pull_adds_its_torque_to_its_motors_for_the_hold_time():
    steps = Steps(0.03, [(0, 0.04), ([1, 2], 0.05)], held_out=[(2, 0.06)], hold=2.0, rest=1.0)
    rows = run(steps)
    assert steps.duration == 10.0  # a rest, then three pulls of 2 s, each followed by a rest
    base = [0.03, 0.03, 0.03]
    np.testing.assert_allclose(torque(rows, 0.5), base)  # the first rest
    np.testing.assert_allclose(torque(rows, 1.0), [0.07, 0.03, 0.03])  # the first pull starts
    np.testing.assert_allclose(torque(rows, 2.99), [0.07, 0.03, 0.03])
    np.testing.assert_allclose(torque(rows, 3.0), base)  # and ends after 2 s
    np.testing.assert_allclose(torque(rows, 4.0), [0.03, 0.08, 0.08])
    np.testing.assert_allclose(torque(rows, 6.5), base)
    np.testing.assert_allclose(torque(rows, 7.5), [0.03, 0.03, 0.09])
    np.testing.assert_allclose(torque(rows, 9.5), base)


def test_the_log_marks_the_training_steps():
    steps = Steps(0.03, [(0, 0.04), (1, 0.04)], held_out=[(2, 0.04)], hold=2.0, rest=1.0)
    rows = run(steps, T=steps.duration + 3.0)  # three seconds past the end
    train, t = rows["train"].ravel(), rows["t"].ravel()
    assert train[t < 7.0 - 1e-6].all()  # the first rest, the two training pulls and their rests
    assert not train[t > 7.0 + 1e-6].any()  # the held-out pull, its rest and what follows


def test_without_held_out_pulls_every_step_trains():
    assert run(Steps(0.03, [(0, 0.04)]), T=10.0)["train"].all()


def test_no_pulls_hold_the_baseline_for_ever():
    rows = run(Steps([0.01, 0.02, 0.03]), T=3.0)
    np.testing.assert_allclose(rows["motor_torque"], np.tile([0.01, 0.02, 0.03], (300, 1)))
    assert rows["train"].all()


def test_the_script_starts_when_the_run_does():
    plant = Still()
    plant.t = 100.0  # a plant whose clock was running before the run
    rows = run(Steps(0.0, [(0, 1.0)], hold=1.0, rest=1.0), T=3.0, plant=plant)
    assert torque(rows, 0.5)[0] == 0.0
    assert torque(rows, 1.5)[0] == 1.0


K, D = 0.6 * helyx.SIM_STIFFNESS, 0.8 * helyx.SIM_DAMPING


@cache
def experiment_on_the_arm():
    """Steps on the simulated arm: nine training pulls and two held-out ones."""
    plant = vmc.sim.ModelPlant(helyx.add_dynamics(helyx.arm("145-145-145"), stiffness=K, damping=D))
    clock = vmc.sim.SimClock(1 / 330)
    vmc.sim.run(plant, Steps(0.03), clock, T=3.0)  # settle at the baseline
    steps = Steps(0.03, [(m, 0.04) for m in range(9)], [(0, 0.06), (4, 0.06)], hold=1.0, rest=1.0)
    return vmc.sim.run(plant, steps, clock, T=steps.duration)


def known_arm():
    known = helyx.arm("145-145-145")
    known.add("gravity", vmc.Gravity(known))
    return known


def test_the_log_of_a_run_goes_straight_to_the_fit():
    K_fit, D_fit = fit_stiffness_damping(known_arm(), [experiment_on_the_arm()], smoothing=11)
    np.testing.assert_allclose(K_fit, K, rtol=0.05)
    np.testing.assert_allclose(D_fit, D, rtol=0.03)


def test_the_fit_leaves_out_the_held_out_steps():
    known, rows = known_arm(), experiment_on_the_arm().arrays()
    train = rows["train"].ravel() > 0
    assert not train.all()  # there are held-out samples
    fit = fit_stiffness_damping(known, [rows], smoothing=11)

    def fit_with(u):
        return fit_stiffness_damping(known, [{**rows, "motor_torque": u}], smoothing=11)

    spoilt = np.where(train[:, None], rows["motor_torque"], 1.0)  # no model explains this torque
    for x, y in zip(fit, fit_with(spoilt), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-3)  # on the held-out steps, nothing changes
    spoilt = np.where(train[:, None], 1.0, rows["motor_torque"])
    assert abs(fit_with(spoilt)[0] / fit[0] - 1).max() > 0.5  # on the training ones, it follows


def arm_with(K, D):
    return helyx.add_dynamics(helyx.arm("145-145-145"), stiffness=K, damping=D)


def test_the_true_model_reproduces_the_held_out_steps():
    result = validate(arm_with(K, D), experiment_on_the_arm())
    np.testing.assert_allclose(result["rms"], 0.0, atol=1e-9)
    np.testing.assert_allclose(result["vaf"], 1.0, atol=1e-9)
    assert result["t"].shape == (len(result["q"]),) and result["q"].shape[1] == 9


def test_the_model_runs_from_the_first_to_the_last_held_out_sample():
    rows = experiment_on_the_arm().arrays()
    train = rows["train"].ravel() > 0
    train[-100:] = True  # the run ends with training samples
    held = np.flatnonzero(~train)
    result = validate(arm_with(1.2 * K, D), {**rows, "train": train})  # a model that soon leaves
    np.testing.assert_array_equal(result["q"][0], rows["q"][held[0]])  # the run, where it starts
    np.testing.assert_allclose(result["t"], rows["t"].ravel()[held[0] : held[-1] + 1])


def test_a_stiffer_model_is_further_from_the_held_out_steps():
    exact = validate(arm_with(K, D), experiment_on_the_arm())
    stiffer = validate(arm_with(1.2 * K, D), experiment_on_the_arm())
    assert stiffer["rms"].max() > 1e-3  # [m]
    assert stiffer["vaf"].max() < 0.995  # and it explains less of every coordinate's motion
    assert (stiffer["rms"] > 1e3 * exact["rms"]).all()


def test_the_fitted_model_explains_the_held_out_steps():
    log = experiment_on_the_arm()
    known = known_arm()
    K_fit, D_fit = fit_stiffness_damping(known, [log], smoothing=11)
    result = validate(arm_with(K_fit, D_fit), log)
    assert result["vaf"].min() > 0.95
    assert result["rms"].max() < 1e-3  # [m]


def test_only_the_held_out_samples_count():
    rows = dict(experiment_on_the_arm().arrays())
    train = rows["train"].ravel() > 0
    held = np.flatnonzero(~train)
    assert held.size > 20
    marked = train.copy()
    marked[held[10:20]] = True  # some samples inside the held-out span trained
    q = rows["q"].copy()
    q[held[10:20]] += 1.0  # and are far from what the model does
    result = validate(arm_with(K, D), {**rows, "train": marked, "q": q})
    np.testing.assert_allclose(result["rms"], 0.0, atol=1e-9)


def test_a_run_without_held_out_steps_cannot_be_validated():
    rows = experiment_on_the_arm().arrays()
    with pytest.raises(ValueError, match="no held-out"):
        validate(arm_with(K, D), {**rows, "train": np.ones(len(rows["t"]))})
