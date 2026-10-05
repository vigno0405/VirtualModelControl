"""A simulated session: the simulation in a worker thread, a person at the window, and what they
did kept as a schedule that runs again exactly."""

import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.interactive import Session, Window

DT = 1 / 330
GOAL = "ctrl.reach.goal"
STIFFNESS = "ctrl.reach.stiffness"
SIGNALS = ("t", "motor_torque", "law_torque", "motor_position", "motor_velocity", "q", "v")


@pytest.fixture(autouse=True)
def close_windows():
    yield
    plt.close("all")


def wait_for(condition, timeout=30.0):
    start = time.monotonic()
    while not condition():
        assert time.monotonic() - start < timeout, "the simulation did not get there"
        time.sleep(0.002)


def clock(session):
    """The simulated time of the latest step."""
    return session.interactive.last[0]


def after(session, steps):
    """Let the worker thread take ``steps`` more steps."""
    wait_for(lambda: session.interactive.last is not None)
    target = clock(session) + steps * DT - 1e-9
    wait_for(lambda: clock(session) >= target)
    session.window.refresh()  # the window draws from this thread while the other simulates


def test_a_session_simulates_in_a_worker_thread_until_it_is_stopped(experiment):
    session = Session.from_experiment(experiment, goals=[GOAL], speed=1000.0)
    session.start()
    after(session, 40)
    log = session.stop()
    t = log.arrays()["t"].ravel()
    assert len(t) >= 40 and np.allclose(np.diff(t), DT)
    assert session.log is log and log.meta["params"]  # a log like any run's
    assert session.stop() is log  # stopping twice is harmless


def test_what_was_done_by_hand_runs_again_exactly_from_the_saved_configuration(
    make_experiment, tmp_path
):
    experiment = make_experiment()
    session = Session.from_experiment(
        experiment, goals=[GOAL], sliders={STIFFNESS: (0.0, 600.0)}, speed=1000.0
    )
    session.recorder.min_dt = 0.0  # keep every value: the replay is then exact
    session.window.keyboard.step = 0.04  # [m] a move the arm follows visibly
    session.controls.toggle_recording()  # record from the first step
    session.start()
    after(session, 30)
    session.window.keyboard.press("right")
    after(session, 20)
    session.window.sliders[STIFFNESS].set_val(450.0)
    after(session, 20)
    session.window.keyboard.press("up")
    session.window.keyboard.press("pageup")
    after(session, 20)
    session.window.radio.set_active(1)  # the gentle controller
    after(session, 100)
    log = session.stop()

    rows = log.arrays()
    steps = len(rows["t"])
    assert np.ptp(rows["q"], axis=0).max() > 1e-3
    path = session.recorder.save(experiment, tmp_path / "session.yaml")
    spec = vmc.config.files.read(path)
    spec["experiment"]["duration"] = steps * DT  # as long as the session was
    again = vmc.config.load(vmc.config.files.write(spec, path)).run().arrays()
    assert len(again["t"]) == steps
    for name in SIGNALS:
        np.testing.assert_array_equal(again[name], rows[name], err_msg=name)

    kinds = [e.get("param", "swap") for e in spec["experiment"]["schedule"]]
    assert sorted(kinds) == sorted([GOAL, STIFFNESS, "swap"])


def test_stopping_a_paused_session_does_not_hang(experiment):
    session = Session.from_experiment(experiment, goals=[GOAL], speed=1000.0)
    session.start()
    after(session, 10)
    session.window.pause.ax.figure.canvas.draw()
    session.controls.toggle_pause()
    time.sleep(0.05)
    start = time.monotonic()
    assert len(session.stop().arrays()["t"]) >= 10
    assert time.monotonic() - start < 2.0


def test_an_error_in_the_simulation_comes_out_of_stop(experiment, monkeypatch):
    session = Session.from_experiment(experiment, goals=[GOAL], speed=1000.0)

    def broken(dt):
        raise RuntimeError("the plant failed")

    monkeypatch.setattr(session.plant, "advance", broken)
    session.start()
    wait_for(lambda: not session._thread.is_alive())  # the worker died of it
    with pytest.raises(RuntimeError, match="the plant failed"):
        session.stop()


def test_run_opens_the_window_and_returns_the_log_when_it_is_closed(experiment, monkeypatch):
    session = Session.from_experiment(experiment, goals=[GOAL], speed=1000.0)

    def closed_after_a_while(window):
        after(session, 25)

    monkeypatch.setattr(Window, "show", closed_after_a_while)
    log = session.run()
    assert len(log.arrays()["t"]) >= 25
