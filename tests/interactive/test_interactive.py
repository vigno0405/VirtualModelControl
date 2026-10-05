"""The wrapper that applies hand-made changes, the energy they give, and the schedule that records
them: a recorded session runs again exactly."""

import threading
import time

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.interactive import Interactive, Recorder

DT = 1 / 330
GOAL = "ctrl.reach.goal"
SIGNALS = ("t", "motor_torque", "law_torque", "motor_position", "motor_velocity", "q", "v")


class Script:
    """A person at the window, as a script: at the given steps it leaves changes in the controls."""

    def __init__(self, interactive, plan):
        self.interactive, self.plan, self.n = interactive, plan, 0

    def reset(self, t, meas=None, z0=None):
        self.n = 0
        self.interactive.reset(t, meas, z0)

    def step(self, t, meas):
        for action in self.plan.get(self.n, []):
            action(self.interactive.controls)
        self.n += 1
        return self.interactive.step(t, meas)

    def __getattr__(self, name):
        return getattr(self.interactive, name)


def interactive_of(experiment, **kwargs):
    swaps = {k: c for k, c in experiment.controllers.items() if k != experiment.name}
    return Interactive(experiment.controller, name=experiment.name, swaps=swaps, **kwargs)


def play(experiment, plan, T=1.0, clock=None, record=(), **kwargs):
    """Run the experiment's simulation with the changes of ``plan`` made at their steps."""
    interactive = interactive_of(experiment, **kwargs)
    experiment.plant.reset()
    clock = clock or vmc.sim.SimClock(DT)
    log = vmc.sim.run(experiment.plant, Script(interactive, plan), clock, T, record=record)
    return interactive, log


def test_a_change_is_applied_just_before_the_step_that_follows_it(experiment):
    goal = np.array([0.05, 0.0, 0.40])
    plan = {10: [lambda c: c.set(GOAL, goal)]}
    _, log = play(experiment, plan, T=0.1, record="params")
    held = log.arrays()["param/" + GOAL]
    np.testing.assert_array_equal(held[:10], [[0.0, 0.0, 0.435]] * 10)
    np.testing.assert_array_equal(held[10:], [goal] * (len(held) - 10))
    rows = log.arrays()
    assert not np.array_equal(rows["motor_torque"][9], rows["motor_torque"][10])  # and it acted


class SetBefore:
    """The definition of 'just before the step': set the values, then step the controller."""

    def __init__(self, controller, at, values):
        self.controller, self.at, self.values, self.n = controller, at, values, 0

    def reset(self, t, meas=None, z0=None):
        self.n = 0
        self.controller.reset(t, meas, z0)

    def step(self, t, meas):
        if self.n == self.at:
            self.controller.set(self.values)
        self.n += 1
        return self.controller.step(t, meas)

    def __getattr__(self, name):
        return getattr(self.controller, name)


def test_a_change_acts_exactly_as_setting_it_on_the_controller_before_the_step(make_experiment):
    values = {GOAL: np.array([0.05, 0.0, 0.40]), "ctrl.damp.damping": 9.0}
    at_step_10 = [lambda c: [c.set(name, value) for name, value in values.items()]]
    _, by_hand = play(make_experiment(), {10: at_step_10}, T=0.1)
    plain = make_experiment()
    plain.plant.reset()
    clock = vmc.sim.SimClock(DT)
    direct = vmc.sim.run(plain.plant, SetBefore(plain.controller, 10, values), clock, T=0.1)
    for name in SIGNALS:
        np.testing.assert_array_equal(by_hand.arrays()[name], direct.arrays()[name], err_msg=name)


def test_a_value_reaches_every_controller_that_has_the_param_live(experiment):
    plan = {5: [lambda c: c.set("gentle.reach.stiffness", 80.0)]}  # ctrl is the one running
    play(experiment, plan, T=0.1)
    gentle = experiment.controllers["gentle"].live_params()
    assert gentle["gentle.reach.stiffness"] == 80.0
    assert experiment.controllers["ctrl"].live_params()["ctrl.reach.stiffness"] == 300.0


def test_a_swap_blends_the_other_controller_in(experiment):
    plan = {5: [lambda c: c.swap("gentle", 0.05)]}
    interactive, log = play(experiment, plan, T=0.2, record="elements")
    rows = log.arrays()
    first, second = (
        np.isfinite(rows[f"element/{n}.reach/force"][:, 0]) for n in ("ctrl", "gentle")
    )
    assert first[:5].all() and not first[-1] and second[-1] and not second[0]
    assert (first ^ second).all()  # one controller's elements at every step
    assert interactive.selected == "gentle"


def test_stopping_ends_a_paced_run_with_the_log_so_far(experiment):
    plan = {7: [lambda c: c.stop()]}
    _, log = play(experiment, plan, T=None, clock=vmc.sim.SimClock(DT, speed=1e6))
    assert len(log.arrays()["t"]) == 7


def test_a_pause_holds_a_simulation_until_it_is_resumed(experiment):
    interactive = interactive_of(experiment, pausable=True)
    controls = interactive.controls
    plan = {3: [lambda c: setattr(c, "paused", True)]}
    experiment.plant.reset()
    timer = threading.Timer(0.3, controls.toggle_pause)
    timer.start()
    start = time.monotonic()
    log = vmc.sim.run(experiment.plant, Script(interactive, plan), vmc.sim.SimClock(DT), 0.1)
    assert time.monotonic() - start >= 0.28 and len(log.arrays()["t"]) == 33
    timer.join()


def test_a_robot_node_is_not_paused_by_a_flag_it_did_not_ask_for(experiment):
    plan = {3: [lambda c: setattr(c, "paused", True)]}  # not pausable: a real robot must not stall
    _, log = play(experiment, plan, T=0.1)
    assert len(log.arrays()["t"]) == 33


def test_the_energy_that_changes_give_the_controller_is_counted(make_experiment):
    old, new = np.array([0.0, 0.0, 0.435]), np.array([0.08, 0.0, 0.40])
    moved = {10: [lambda c: c.set(GOAL, new)]}
    stiffer = {50: [lambda c: c.set("ctrl.reach.stiffness", 600.0)]}
    first, log = play(make_experiment(), moved, T=0.3, record="elements")
    both, _ = play(make_experiment(), moved | stiffer, T=0.3)
    y = log.arrays()["element/ctrl.reach/y"]  # tip minus goal, step by step

    # A goal moved from `old` to `new` changes the spring's energy 1/2 K |y|^2 by
    # 1/2 K (|y + old - new|^2 - |y|^2), with y the deflection the change finds.
    after = y[9] + old - new
    assert first.injected == pytest.approx(0.5 * 300.0 * (after @ after - y[9] @ y[9]), rel=1e-9)
    # A spring made stiffer by dK gains 1/2 dK |y|^2.
    assert both.injected - first.injected == pytest.approx(0.5 * 300.0 * (y[49] @ y[49]), rel=1e-9)
    assert first.injected > 0.1 and both.injected > first.injected
    first.reset(0.0)
    assert first.injected == 0.0


def test_the_wrapper_is_the_controller_for_everything_else(experiment):
    interactive = interactive_of(experiment)
    assert interactive.compiled is experiment.controllers["ctrl"].compiled
    assert interactive.unit("ctrl.reach.stiffness") == "N/m"
    assert sorted(interactive.controls.values) == sorted(
        experiment.controllers["ctrl"].live_params()
        | experiment.controllers["gentle"].live_params()
    )
    assert interactive.controls.controllers == ["ctrl", "gentle"]


def test_the_recorder_keeps_values_and_swaps_with_times_from_its_start():
    recorder = Recorder(min_dt=0.0)
    recorder.start(2.0, {"a": np.array([1.0, 2.0]), "b": np.array(7.0)})
    recorder.applied(2.5, {"a": np.array([3.0, 4.0])}, [])
    recorder.applied(3.0, {}, [("gentle", 0.5)])
    recorder.applied(3.25, {"a": np.array([5.0, 6.0]), "b": np.array(8.0)}, [])
    recorder.stop()
    assert recorder.schedule() == [
        {
            "param": "a",
            "points": [[0.0, [1.0, 2.0]], [0.5, [3.0, 4.0]], [1.25, [5.0, 6.0]]],
            "interpolation": "step",
        },
        {"param": "b", "points": [[0.0, 7.0], [1.25, 8.0]], "interpolation": "step"},
        {"swap": "gentle", "at": 1.0, "duration": 0.5},
    ]


def test_values_closer_than_min_dt_are_merged_and_the_last_one_survives():
    recorder = Recorder(min_dt=0.02)
    recorder.start(0.0, {"a": np.array(0.0)})
    for t, value in (
        (0.0, 1.0),
        (0.005, 2.0),
        (0.010, 3.0),
        (0.025, 4.0),
        (0.030, 5.0),
        (0.035, 6.0),
    ):
        recorder.applied(t, {"a": np.array(value)}, [])
    recorder.stop()
    (entry,) = recorder.schedule()
    assert entry["points"] == [[0.0, 1.0], [0.025, 4.0], [0.035, 6.0]]  # 0.0 was the first change


def test_recording_from_the_middle_of_a_run_counts_time_from_there(experiment):
    plan = {
        30: [lambda c: c.toggle_recording()],
        40: [lambda c: c.set(GOAL, [0.05, 0.0, 0.4]), lambda c: c.swap("gentle", 0.1)],
    }
    interactive, _ = play(experiment, plan, T=0.2, recorder=Recorder(min_dt=0.0))
    goal, swap = interactive.recorder.schedule()
    assert goal["points"][0] == [0.0, [0.0, 0.0, 0.435]]  # what it was when recording began
    assert goal["points"][1][0] == pytest.approx(10 * DT) and goal["points"][1][1] == [
        0.05,
        0.0,
        0.4,
    ]
    assert swap["at"] == pytest.approx(10 * DT) and swap["duration"] == 0.1


SESSION = {
    0: [lambda c: c.toggle_recording()],
    20: [lambda c: c.set(GOAL, [0.06, 0.0, 0.41])],
    45: [
        lambda c: c.set("ctrl.reach.stiffness", 500.0),
        lambda c: c.nudge(GOAL, [0.0, 0.02, 0.0]),
    ],
    80: [lambda c: c.swap("gentle", 0.1)],
    120: [lambda c: c.set("ctrl.damp.damping", 8.0)],
    250: [lambda c: c.swap("ctrl", 0.2)],
}


def test_a_recorded_session_runs_again_exactly_from_the_saved_configuration(experiment, tmp_path):
    interactive, log = play(experiment, SESSION, recorder=Recorder(min_dt=0.0))
    path = interactive.recorder.save(experiment, tmp_path / "session.yaml")

    again = vmc.config.load(path).run().arrays()  # a plain experiment: nobody at the window
    rows = log.arrays()
    for name in SIGNALS:
        np.testing.assert_array_equal(again[name], rows[name], err_msg=name)
    assert np.ptp(rows["q"], axis=0).max() > 1e-3  # the arm moved


def test_saving_adds_to_the_schedule_the_configuration_already_has(experiment, tmp_path):
    spec = experiment.to_dict()
    own = {"param": "ctrl.reach.stiffness", "points": [[0.0, 250.0]], "interpolation": "step"}
    spec["experiment"]["schedule"] = [own]
    recorder = Recorder()
    recorder.start(0.0, {GOAL: np.zeros(3)})
    recorder.applied(0.1, {GOAL: np.array([0.1, 0.0, 0.4])}, [])
    recorder.stop()
    for source in (
        spec,
        vmc.config.files.write(spec, tmp_path / "own.yaml"),
        vmc.config.load(spec),
    ):
        saved = vmc.config.files.read(recorder.save(source, tmp_path / "saved.yaml"))
        assert saved["experiment"]["schedule"][0] == own
        assert saved["experiment"]["schedule"][1]["param"] == GOAL
        assert len(saved["experiment"]["schedule"]) == 2
