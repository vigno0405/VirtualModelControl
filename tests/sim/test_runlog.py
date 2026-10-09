"""Run logs: aligned steps, whole-or-nothing saving, loading, CSV export and comparison."""

from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx
from virtualmodelcontrol.sim import RunLog, compare

nan = np.nan


def example_log():
    """Three steps: a vector, a matrix, and names that come and go."""
    log = RunLog(meta={"library": "x", "params": {"k": {"value": [1.0, 2.0], "unit": "N/m"}}})
    log.info["steps"] = 3
    log.step(t=0.0, q=[1.0, 2.0], gain=[[1.0, 2.0], [3.0, 4.0]], tag=7.0)
    log.step(t=0.5, q=[3.0, 4.0], gain=[[5.0, 6.0], [7.0, 8.0]])
    log.step(t=1.0, q=[5.0, 6.0], gain=[[0.0, 0.0], [0.0, 0.5]], late=[9.0])
    return log


def test_every_signal_keeps_one_row_per_step():
    rows = example_log().arrays()
    assert {name: values.shape for name, values in rows.items()} == {
        "t": (3, 1),
        "q": (3, 2),
        "gain": (3, 2, 2),
        "tag": (3, 1),  # given at the first step only
        "late": (3, 1),  # given at the last step only
    }
    np.testing.assert_array_equal(rows["tag"], [[7.0], [nan], [nan]])
    np.testing.assert_array_equal(rows["late"], [[nan], [nan], [9.0]])


def test_a_step_copies_what_it_is_given():
    log, state = RunLog(), np.zeros(2)
    for _ in range(3):
        state += 1.0  # updated in place, as a controller's state is
        log.step(z=state)
    np.testing.assert_array_equal(log.arrays()["z"], [[1, 1], [2, 2], [3, 3]])


def test_a_saved_log_loads_back_whole(tmp_path):
    log = example_log()
    log.step(**{"element/ctrl.reach/y": [1.0, 2.0, 3.0], "t": 1.5})  # names with slashes
    path = log.save(tmp_path / "run")  # the suffix is added
    assert path == tmp_path / "run.npz"
    assert list(tmp_path.iterdir()) == [path]  # no temporary file left

    back = RunLog.load(path)
    assert back.meta == log.meta and back.info == log.info
    rows, again = log.arrays(), back.arrays()
    assert list(again) == list(rows)
    for name in rows:
        np.testing.assert_array_equal(again[name], rows[name], err_msg=name)  # NaN rows too


def test_metadata_may_hold_numpy_values(tmp_path):
    log = RunLog(meta={"gain": np.array([1.0, 2.0]), "n": np.int64(3)})
    log.step(t=0.0)
    back = RunLog.load(log.save(tmp_path / "run.npz"))
    assert back.meta == {"gain": [1.0, 2.0], "n": 3}


def test_a_taken_name_is_refused_unless_overwritten(tmp_path):
    first, second = example_log(), RunLog()
    second.step(t=0.0, q=[0.0])
    path = first.save(tmp_path / "run.npz")
    with pytest.raises(FileExistsError, match="overwrite"):
        second.save(path)
    assert sorted(RunLog.load(path).arrays()) == sorted(first.arrays())  # kept as it was
    second.save(path, overwrite=True)
    assert list(RunLog.load(path).arrays()) == ["t", "q"]
    assert list(tmp_path.iterdir()) == [path]


def test_a_failed_save_leaves_the_old_file_and_no_part(tmp_path, monkeypatch):
    log = example_log()
    path = log.save(tmp_path / "run.npz")
    before = path.read_bytes()

    def half_written(file, **arrays):
        Path(file).write_bytes(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(np, "savez_compressed", half_written)
    with pytest.raises(OSError, match="disk full"):
        log.save(path, overwrite=True)
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


def test_a_metadata_that_cannot_be_stored_writes_nothing(tmp_path):
    log = RunLog(meta={"object": object()})
    log.step(t=0.0)
    with pytest.raises(TypeError, match="metadata"):
        log.save(tmp_path / "run.npz")
    assert list(tmp_path.iterdir()) == []


def test_a_newer_or_foreign_file_is_refused(tmp_path):
    np.savez(tmp_path / "foreign.npz", t=np.zeros(3))
    with pytest.raises(ValueError, match="not a run log"):
        RunLog.load(tmp_path / "foreign.npz")

    path = example_log().save(tmp_path / "run.npz")
    with np.load(path) as data:
        contents = dict(data)
    contents["_schema"] = np.array(99)
    np.savez(tmp_path / "newer.npz", **contents)
    with pytest.raises(ValueError, match="schema 99, newer than"):
        RunLog.load(tmp_path / "newer.npz")


def test_csv_has_one_column_per_entry_and_exact_numbers(tmp_path):
    log = RunLog()
    log.step(t=0.0, q=[0.1 + 0.2, 2.0], gain=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], tag=7.0)
    log.step(t=0.5, q=[3.0, 4.0], gain=[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    lines = log.to_csv(tmp_path / "run.csv").read_text().splitlines()
    gain = [f"gain_{i}_{j}" for i in range(2) for j in range(3)]
    assert lines[0] == ",".join(["t", "q_0", "q_1", *gain, "tag"])
    first, second = (line.split(",") for line in lines[1:])
    assert [float(x) for x in first] == [0.0, 0.1 + 0.2, 2.0, 1, 2, 3, 4, 5, 6, 7.0]  # exact
    assert second[-1] == "nan" and second[0] == "0.5"
    table = np.genfromtxt(tmp_path / "run.csv", delimiter=",", names=True)
    np.testing.assert_array_equal(table["gain_1_0"], [4.0, 10.0])
    np.testing.assert_array_equal(table["gain_0_2"], [3.0, 9.0])


def straight(t, offset=0.0, slope=1.0):
    log = RunLog()
    for time in t:
        log.step(t=time, x=[slope * time + offset, 2.0 * slope * time + offset], other=1.0)
    return log


def test_compare_gives_the_largest_and_the_rms_difference_signal_by_signal():
    a = straight(np.linspace(0.0, 1.0, 11))
    b = straight(np.linspace(0.0, 1.0, 5), offset=0.25)  # coarser times, a constant offset
    gap = compare(a, b, names=["x"])
    assert list(gap) == ["x"]
    assert gap["x"]["max"] == pytest.approx(0.25) and gap["x"]["rms"] == pytest.approx(0.25)
    assert sorted(compare(a, b)) == ["other", "x"]  # by default every common signal but the times
    assert compare(a, b)["other"] == {"max": 0.0, "rms": 0.0}
    assert compare(a, a)["x"] == {"max": 0.0, "rms": 0.0}


def test_compare_reads_the_other_run_where_it_has_one_and_skips_nan_steps():
    a = straight(np.linspace(0.0, 1.0, 11))
    b = straight(np.linspace(0.3, 0.7, 5), offset=0.5)  # only the middle of a's run
    assert compare(a, b, names=["x"])["x"]["max"] == pytest.approx(0.5)  # not the edges of a
    c = straight(np.linspace(0.0, 1.0, 11), offset=0.1)
    c.rows["x"][4] = np.full(2, nan)  # a missing step is left out, not counted as a gap
    gap = compare(a, c, names=["x"])["x"]
    assert gap["max"] == pytest.approx(0.1) and gap["rms"] == pytest.approx(0.1)


def test_the_library_reads_runs_of_its_own_simulator(tmp_path):
    arm = helyx.add_dynamics(helyx.arm("145-145-145"))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    log = vmc.sim.run(vmc.sim.ModelPlant(arm), controller, vmc.sim.SimClock(1 / 330), T=0.05)
    back = RunLog.load(log.save(tmp_path / "damped"))
    assert back.meta == log.meta and back.meta["library"] == vmc.__version__
    for name, values in log.arrays().items():
        np.testing.assert_array_equal(back.arrays()[name], values, err_msg=name)


def test_crop_keeps_the_steps_of_a_time_window_and_leaves_the_log_as_it_was():
    log = example_log()
    part = log.crop(0.25, 1.0)
    rows = part.arrays()
    np.testing.assert_array_equal(rows["t"].ravel(), [0.5, 1.0])
    np.testing.assert_array_equal(rows["q"], [[3.0, 4.0], [5.0, 6.0]])
    assert rows["tag"].shape == (2, 1) and np.isnan(rows["tag"]).all()  # the names stay, aligned
    assert part.info == log.info and part.meta == log.meta
    part.info["steps"] = 0  # a copy: the original is not touched
    assert log.info["steps"] == 3 and len(log.arrays()["t"]) == 3
    np.testing.assert_array_equal(
        log.crop(t_max=0.5).arrays()["t"].ravel(), [0.0, 0.5]
    )  # open ends
    np.testing.assert_array_equal(log.crop(t_min=0.6).arrays()["t"].ravel(), [1.0])
    assert len(log.crop().arrays()["t"]) == 3 and len(log.crop(2.0, 3.0).rows["t"]) == 0
    np.testing.assert_array_equal(
        log.crop(0.5, 0.5).arrays()["t"].ravel(), [0.5]
    )  # both ends count


def test_crop_with_an_open_end_takes_every_time_on_that_side():
    log = RunLog()
    for t in (-1.0, 0.0, 5.0):
        log.step(t=t, q=[t])
    np.testing.assert_array_equal(log.crop(t_max=0.0).arrays()["t"].ravel(), [-1.0, 0.0])
    np.testing.assert_array_equal(log.crop(t_min=0.0).arrays()["t"].ravel(), [0.0, 5.0])
