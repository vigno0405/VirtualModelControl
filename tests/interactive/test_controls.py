"""The mailbox: values and swaps asked for, taken once, from any thread."""

import threading

import numpy as np
import pytest

from virtualmodelcontrol.interactive import Controls


def controls():
    return Controls({"goal": [0.0, 0.0, 0.4], "k": 300.0}, ["ctrl", "gentle"])


def test_the_latest_value_is_what_the_loop_takes_and_what_others_see():
    c = controls()
    c.set("goal", [0.1, 0.0, 0.4])
    c.set("goal", [0.2, 0.0, 0.4])  # before the loop took the first
    np.testing.assert_array_equal(c.values["goal"], [0.2, 0.0, 0.4])
    pending, swaps = c.take()
    assert list(pending) == ["goal"] and swaps == []
    np.testing.assert_array_equal(pending["goal"], [0.2, 0.0, 0.4])
    assert c.take() == ({}, [])  # taken once
    np.testing.assert_array_equal(c.values["goal"], [0.2, 0.0, 0.4])  # still the latest


def test_a_nudge_adds_to_the_latest_value_even_before_it_was_taken():
    c = controls()
    c.nudge("goal", [0.01, 0.0, 0.0])
    c.nudge("goal", [0.01, 0.0, -0.1])
    np.testing.assert_allclose(c.take()[0]["goal"], [0.02, 0.0, 0.3])
    c.nudge("k", 50.0)  # a scalar
    assert c.take()[0]["k"] == pytest.approx(350.0)


def test_values_are_copies():
    c = controls()
    c.values["goal"][0] = 9.0
    assert c.values["goal"][0] == 0.0
    given = np.array([1.0, 2.0, 3.0])
    c.set("goal", given)
    given[0] = -5.0
    assert c.values["goal"][0] == 1.0


def test_unknown_names_and_wrong_sizes_are_refused_where_they_are_made():
    c = controls()
    with pytest.raises(KeyError, match="not a live Param here"):
        c.set("nope", 1.0)
    with pytest.raises(KeyError, match="not a live Param here"):
        c.nudge("nope", 1.0)
    with pytest.raises(ValueError):
        c.set("goal", [1.0, 2.0])
    with pytest.raises(KeyError, match="no controller named 'nope'"):
        c.swap("nope")
    assert c.take() == ({}, [])  # nothing half-done was left behind


def test_swaps_wait_in_order_and_the_flags_toggle():
    c = controls()
    c.swap("gentle", 0.5)
    c.swap("ctrl")
    assert c.take()[1] == [("gentle", 0.5), ("ctrl", 1.0)]
    assert [c.toggle_pause(), c.toggle_pause()] == [True, False]
    assert [c.toggle_recording(), c.toggle_recording()] == [True, False]
    assert not c.stopped
    c.stop()
    assert c.stopped


def test_many_threads_can_leave_changes_at_once():
    c = Controls({"count": 0.0})

    def worker():
        for _ in range(500):
            c.nudge("count", 1.0)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert c.values["count"][()] == 2000.0  # no change was lost
