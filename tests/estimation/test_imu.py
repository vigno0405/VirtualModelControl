"""The IMU filter: against the lab's on a synthetic run, and against the arm's own kinematics."""

from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.estimation import ImuFilter
from virtualmodelcontrol.robots import helyx

DATA = np.load(Path(__file__).parents[1] / "data" / "imu.npz")
SITES = ("base", "seg1", "seg2", "tip")
G = 9.81


def labs_filter():
    return ImuFilter(
        helyx.arm("290-145-145"),
        mounts=DATA["mounts"],
        kp=float(DATA["kp"]),
        acc_tol=float(DATA["acc_tol"]),
        gravity=float(DATA["gravity"]),
    )


def labs_run():
    """The library's filter over the lab's run: calibrate, move, reset, hold."""
    f = labs_filter()
    f.calibrate(DATA["cal_gyro"][None], DATA["cal_acc"][None])
    start, q, v = f.q, [], []
    for k in range(len(DATA["dt"])):
        if k == int(DATA["n_move"]):
            f.reset()
        d, dd = f.update(DATA["gyro"][k], DATA["acc"][k], float(DATA["dt"][k]))
        q.append(d), v.append(dd)
    return f, start, np.array(q), np.array(v)


def test_the_calibration_gives_the_labs_gyro_bias_and_starting_pose():
    f = labs_filter()
    f.calibrate(DATA["cal_gyro"][None], DATA["cal_acc"][None])
    np.testing.assert_allclose(f.gyro_bias, DATA["cal_bias"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(f.q, DATA["cal_q"], rtol=1e-6, atol=1e-10)
    np.testing.assert_array_equal(f.v, 0.0)
    assert np.abs(f.q).max() > 5e-3  # it found the bend of the still arm


def test_the_calibration_uses_the_mean_of_its_samples():
    f = labs_filter()
    e = np.random.default_rng(1).normal(0, 0.01, DATA["cal_gyro"].shape)
    f.calibrate(
        np.stack([DATA["cal_gyro"] - e, DATA["cal_gyro"] + e]),
        np.stack([DATA["cal_acc"] + e, DATA["cal_acc"] - e]),
    )
    np.testing.assert_allclose(f.gyro_bias, DATA["cal_bias"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(f.q, DATA["cal_q"], rtol=1e-6, atol=1e-10)


def test_the_curvatures_and_rates_follow_the_labs_over_the_whole_run():
    _, _, q, v = labs_run()
    np.testing.assert_allclose(q, DATA["q"], rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(v, DATA["v"], rtol=1e-6, atol=1e-8)
    assert np.abs(q).max() > 1e-2 and np.abs(v).max() > 0.1  # a run that moved


def test_the_run_skips_the_correction_where_the_accelerometers_cannot_be_trusted():
    # the lab's IMU 2 reads 1.3 g at steps 60 to 79 and none reads anything at 100 to 104; the
    # gyros carry those steps, as in the lab (the whole run above); a filter that trusts every
    # reading goes another way
    f, g = labs_filter(), labs_filter()
    g.acc_tol = 10.0
    for h in (f, g):
        h.calibrate(DATA["cal_gyro"][None], DATA["cal_acc"][None])
    q_f, q_g = [], []
    for k in range(100):
        args = (DATA["gyro"][k], DATA["acc"][k], float(DATA["dt"][k]))
        q_f.append(f.update(*args)[0]), q_g.append(g.update(*args)[0])
    np.testing.assert_allclose(q_f, DATA["q"][:100], rtol=1e-6, atol=1e-10)
    assert np.abs(np.array(q_g) - DATA["q"][:100]).max() > 5e-4


def test_a_reset_starts_from_the_straight_pose_and_converges_on_a_held_bend():
    f, _, q, _ = labs_run()
    n = int(DATA["n_move"])
    assert np.abs(q[n] - DATA["hold_q"]).max() < np.abs(DATA["hold_q"]).max()  # moved toward it
    assert np.abs(q[-1] - DATA["hold_q"]).max() < np.abs(q[n] - DATA["hold_q"]).max()
    np.testing.assert_array_equal(f.q, q[-1])
    f.reset()
    np.testing.assert_array_equal(f.q, 0.0)
    np.testing.assert_array_equal(f.v, 0.0)
    assert np.abs(f.gyro_bias).max() > 0  # the bias stays


def readings(kin, q, v, mounts=None):
    """What ideal IMUs on the arm's base and section ends read at (q, v): gyro and gravity."""
    gyro, acc = [], []
    for k, site in enumerate(SITES):
        R, Jw = kin.rotation(q, site), kin.angular_jacobian(q, site)
        M = np.eye(3) if mounts is None else mounts[k]
        gyro.append(M.T @ (R.T @ (Jw @ v)))
        acc.append(M.T @ (R.T @ np.array([0.0, 0.0, G])))
    return np.array(gyro), np.array(acc)


def bend(t):
    """(q, v) of a smooth bend of the three sections of the arm, Dl fixed."""
    f = np.array([0.7, 0.5, 0.9, 0.4, 0.6, 0.8])
    idx = [0, 1, 3, 4, 6, 7]
    q, v = np.zeros(9), np.zeros(9)
    q[idx] = 0.015 * np.sin(2 * np.pi * f * t + np.arange(6))
    v[idx] = 0.015 * 2 * np.pi * f * np.cos(2 * np.pi * f * t + np.arange(6))
    return q, v


@pytest.mark.parametrize("radius", [0.03, [0.03, 0.04, 0.05]])
def test_ideal_imus_on_a_moving_arm_give_back_its_curvature_and_rate(radius):
    arm = helyx.arm("290-145-145", section_radius=radius)
    kin, f, dt = vmc.Kinematics(arm), ImuFilter(arm), 0.01
    q0 = bend(0.0)[0]
    still_gyro, still_acc = readings(kin, q0, np.zeros(9))
    f.calibrate(still_gyro[None], still_acc[None])
    seen = np.abs(f.q - q0[f.observed]).max()
    assert seen < 1e-3 and np.abs(q0).max() > 0.01  # the still bend, found from gravity alone
    err_q, err_v = [], []
    for k in range(1, 400):
        q, v = bend(k * dt)
        gyro, acc = readings(kin, q, v)
        est_q, est_v = f.update(gyro, acc, dt)
        err_q.append(est_q - q[f.observed])
        err_v.append(est_v - v[f.observed])
    assert np.abs(err_q).max() < 1.0e-3  # of curvatures up to 15 mm
    assert np.abs(err_v).max() < 2e-3  # the rates come from the gyros, of up to 66 mm/s


def test_mounts_that_turn_the_sensors_are_undone():
    arm = helyx.arm("290-145-145")
    kin, dt = vmc.Kinematics(arm), 0.01
    rng = np.random.default_rng(3)
    turns = []
    for _ in SITES:
        a = rng.uniform(-np.pi, np.pi)
        turns.append(vmc.math.rot_z(a) @ vmc.math.rot_x(0.4 * a))
    f = ImuFilter(arm, mounts=turns)
    still_gyro, still_acc = readings(kin, bend(0.0)[0], np.zeros(9), mounts=turns)
    f.calibrate(still_gyro[None], still_acc[None])
    # the sensors' frames are turned by the transpose of what the filter is told: v_arm = M v_sensor
    assert np.abs(f.q - bend(0.0)[0][f.observed]).max() < 1e-3
    wrong = ImuFilter(arm)
    wrong.calibrate(still_gyro[None], still_acc[None])
    assert np.abs(wrong.q - bend(0.0)[0][wrong.observed]).max() > 3e-3
    err = []
    for k in range(1, 300):
        q, v = bend(k * dt)
        gyro, acc = readings(kin, q, v, mounts=turns)
        err.append(f.update(gyro, acc, dt)[0] - q[f.observed])
    assert np.abs(err).max() < 1.5e-3  # it follows the motion with the mounts undone


def test_a_section_that_only_twists_stays_straight_and_a_wrong_accelerometer_is_not_used():
    f = ImuFilter(helyx.arm("290-145-145"), kp=5.0)
    twist = np.zeros((4, 3))
    twist[1:, 2] = 0.5  # the three tips turn about their backbones
    down = np.tile([0.0, 0.0, G], (4, 1))
    for _ in range(100):
        q, v = f.update(twist, down, 0.01)
    np.testing.assert_allclose(q, 0.0, atol=1e-12)
    np.testing.assert_allclose(v, 0.0, atol=1e-12)
    # IMU 2 reads 2 g and the gravity says the tip IMU is turned over: sections 1 and 2 ignore
    # it, section 0 does not
    tilt = vmc.math.rot_x(0.2)
    acc = down.copy()
    acc[1] = tilt.T @ acc[1]
    acc[2] = 2.0 * acc[2]
    for _ in range(300):
        q, _ = f.update(np.zeros((4, 3)), acc, 0.01)
    assert np.abs(q[:2]).max() > 1e-3
    np.testing.assert_allclose(q[2:], 0.0, atol=1e-12)


def test_an_accelerometer_off_by_exactly_the_tolerance_is_not_trusted():
    # gravity 8 and tolerance 0.25 are exact in binary: a reading of norm 10 is off by 0.25
    tilted = np.array([[0.0, 0.0, 8.0], [0.0, 6.0, 8.0], [0.0, 0.0, 8.0], [0.0, 0.0, 8.0]])
    arm = helyx.arm("290-145-145")
    f, g = ImuFilter(arm, gravity=8.0, acc_tol=0.25), ImuFilter(arm, gravity=8.0, acc_tol=0.26)
    for _ in range(20):
        q, _ = f.update(np.zeros((4, 3)), tilted, 0.01)
        q_trusted, _ = g.update(np.zeros((4, 3)), tilted, 0.01)
    np.testing.assert_array_equal(q, 0.0)
    assert np.abs(q_trusted[:2]).max() > 1e-4


def test_the_defaults_are_the_labs():
    f = ImuFilter(helyx.arm("290-145-145"))
    assert (f.kp, f.acc_tol, f.gravity) == (2.0, 0.1, 9.81)
    np.testing.assert_array_equal(f.gyro_bias, 0.0)


def test_a_tiny_rotation_gives_its_curvature_without_dividing_by_zero():
    f = ImuFilter(helyx.arm("290-145-145"))
    gyro = np.zeros((4, 3))
    gyro[1:, 1] = 1e-8  # the tips turn about y by 1e-10 rad in 0.01 s
    q, _ = f.update(gyro, np.tile([0.0, 0.0, G], (4, 1)), 0.01)
    assert q[0] == pytest.approx(0.03 * 1e-10, rel=1e-6)  # Dx = d times the angle
    np.testing.assert_array_equal(q[1:], 0.0)


def test_a_calibration_starts_from_the_straight_pose_whatever_came_before():
    used, fresh = labs_filter(), labs_filter()
    for k in range(30):  # the filter has followed the lab's run for a while
        used.update(DATA["gyro"][k], DATA["acc"][k], float(DATA["dt"][k]))
    for f in (used, fresh):
        f.calibrate(DATA["cal_gyro"][None], DATA["cal_acc"][None], steps=3)
    np.testing.assert_array_equal(used.q, fresh.q)
    assert np.abs(fresh.q).max() > 1e-4  # three steps took it part of the way


def test_the_filter_reports_what_the_imus_see_and_hands_out_copies():
    arm = helyx.arm("290-145-145")
    f = ImuFilter(arm)
    np.testing.assert_array_equal(f.observed, [0, 1, 3, 4, 6, 7])
    two = helyx.arm("290-145-145", lengths=(0.29, 0.145), tendon_angles=helyx.TENDON_ANGLES[:2])
    np.testing.assert_array_equal(ImuFilter(two).observed, [0, 1, 3, 4])
    assert ImuFilter(two).gyro_bias.shape == (3, 3)
    q, v = f.update(np.zeros((4, 3)), np.tile([0.0, 0.3, G], (4, 1)), 0.01)
    q[:] = 99.0
    v[:] = 99.0
    f.q[:] = 99.0
    assert np.abs(f.q).max() < 1.0 and np.abs(f.v).max() < 1.0


def test_the_filter_needs_a_pcc_arm_and_one_mount_per_imu():
    with pytest.raises(ValueError, match="needs a PCC arm"):
        ImuFilter(helyx.arm("290-145-145").model.space)
    with pytest.raises(ValueError, match=r"4 IMUs need mounts of shape \(4, 3, 3\)"):
        ImuFilter(helyx.arm("290-145-145"), mounts=np.eye(3))
