"""The filter against the lab's: the same 100 steps of a synthetic run, step by step.

The lab's estimator ran with encoders, motion capture and IMUs (in turn), markers and IMU
estimates that drop out and outliers of each sensor; the fixture holds its inputs, its state and
covariance after every step, the sensors its gate rejected and its health bits.
"""

from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.estimation import KalmanFilter, Measurement
from virtualmodelcontrol.robots import helyx

DATA = np.load(Path(__file__).parents[1] / "data" / "kalman.npz")
SENSORS = ("encoder", "mocap", "imu")
BITS = {  # the lab's health bits: (rejected, missing) of each sensor
    "encoder": (1, 0),
    "mocap": (2, 4),
    "imu": (8, 16),
}


def labs_arm_filter():
    """The library's arm with the lab's numbers: lumped masses, physical K and D, no gravity."""
    arm = helyx.arm(
        "290-145-145", masses=DATA["masses"], gravity=(0, 0, 0), efficiency=float(DATA["eta"])
    )
    helyx.add_dynamics(arm, stiffness=DATA["k_struct"], damping=DATA["d_struct"])
    system = vmc.VirtualMechanismSystem(arm, vmc.Mechanism("ctrl"))
    kf = KalmanFilter(system, float(DATA["dt"]), Q=DATA["Q"], gate=float(DATA["gate"]))
    kf.reset(DATA["q0"])
    return kf


def seen_at(kf, k):
    """What the sensors offered at step k, built the way a user would."""
    d, offered = DATA, []
    if d["requested"][k, 0]:
        offered.append(kf.encoder(d["theta"][k], d["theta_dot"][k], d["R_enc"], d["R_enc_dot"]))
    if d["mocap_on"][k]:
        y = d["mocap_y"][k]
        offered.append(Measurement(y[:9], y[9:], d["R_mocap"], d["R_mocap_dot"], name="mocap"))
    if d["imu_on"][k]:
        y, idx = d["imu_y"][k], d["imu_idx"]
        offered.append(
            Measurement(y[:6], y[6:], d["R_imu"], d["R_imu_dot"], observed=idx, name="imu")
        )
    return offered


def distance(kf, m):
    """The innovation distance d2 of a measurement, as the gate sees it."""
    H = m.H(9)
    nu = m.y - H @ np.concatenate([kf.q, kf.v])
    return float(nu @ np.linalg.solve(H @ kf.P @ H.T + m.R, nu))


@pytest.fixture(scope="module")
def run():
    """The library's filter over the lab's run: what it did at each step."""
    kf, steps = labs_arm_filter(), []
    for k in range(len(DATA["tau"])):
        kf.predict(DATA["tau"][k])
        offered = seen_at(kf, k)
        d2 = {m.name: distance(kf, m) for m in offered}
        asked = [s for s, on in zip(SENSORS, DATA["requested"][k], strict=True) if on]
        kf.update(offered, expected=asked)
        steps.append(
            {
                "x": np.concatenate([kf.q, kf.v]),
                "P": kf.P,
                "d2": [d2.get(s, np.nan) for s in SENSORS],
                "rejected": [s in kf.rejected for s in SENSORS],
                "missing": kf.missing,
                "total": kf.rejected_total,
                "y": {m.name: m.y for m in offered},
            }
        )
    return steps


def test_the_state_follows_the_labs_over_the_whole_run(run):
    x = np.array([s["x"] for s in run])
    np.testing.assert_allclose(x, DATA["x"], rtol=1e-7, atol=1e-9)
    assert np.abs(x[:, 9:]).max() > 1e-2  # the velocities it agrees on are not zero


def test_the_covariance_follows_the_labs_in_the_diagonal_and_in_full(run):
    diag = np.array([np.diag(s["P"]) for s in run])
    np.testing.assert_allclose(diag, DATA["P_diag"], rtol=1e-6, atol=1e-11)
    for k, P in zip(DATA["p_steps"], DATA["P_full"], strict=True):
        np.testing.assert_allclose(run[k]["P"], P, rtol=1e-6, atol=1e-11)
    off = np.abs(DATA["P_full"][-1] - np.diag(np.diag(DATA["P_full"][-1]))).max()
    assert off > 1e-7  # the cross terms it agrees on are there


def test_the_gate_rejects_the_labs_sensors_at_the_labs_steps_and_counts_them(run):
    np.testing.assert_array_equal([s["rejected"] for s in run], DATA["rejected"])
    np.testing.assert_array_equal([s["total"] for s in run], DATA["rejected_total"])
    assert DATA["rejected"].sum(0).min() >= 2  # each sensor was rejected at least twice


def test_the_innovation_distance_is_the_labs_and_decides_the_gate(run):
    d2 = np.array([s["d2"] for s in run])
    np.testing.assert_allclose(d2, DATA["d2"], rtol=1e-6, equal_nan=True)
    seen = ~np.isnan(d2)
    np.testing.assert_array_equal(d2[seen] > float(DATA["gate"]), DATA["rejected"][seen])


def test_the_names_of_rejected_and_missing_sensors_make_the_labs_health_bits(run):
    health = []
    for s in run:
        bits = 0
        for i, name in enumerate(SENSORS):
            rejected, missing = BITS[name]
            bits |= rejected * s["rejected"][i] | missing * (name in s["missing"])
        health.append(bits)
    np.testing.assert_array_equal(health, DATA["health"])
    assert {1, 2, 4, 8, 16} <= {b for h in DATA["health"] for b in (1, 2, 4, 8, 16) if h & b}


def test_the_encoder_goes_to_the_state_through_the_labs_map_of_the_motors(run):
    enc = np.array([s["y"].get("encoder", np.full(18, np.nan)) for s in run])
    np.testing.assert_allclose(enc, DATA["enc_y"], rtol=1e-10, atol=1e-14)


def test_the_joseph_form_gives_the_covariance_of_the_updates(run):
    # (I - K H) P is the lab's; with the optimal gain it equals the Joseph form, which holds
    # for any gain, so a wrong gain or a wrong S shows up as a difference
    kf = labs_arm_filter()
    for k in range(0, 100, 11):
        kf.predict(DATA["tau"][k])
        offered = [m for m in seen_at(kf, k) if distance(kf, m) <= float(DATA["gate"])]
        if not offered:
            continue
        H = np.vstack([m.H(9) for m in offered])
        R = np.zeros((H.shape[0],) * 2)
        i = 0
        for m in offered:
            R[i : i + m.y.size, i : i + m.y.size] = m.R
            i += m.y.size
        P = kf.P
        gain = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
        joseph = (np.eye(18) - gain @ H) @ P @ (np.eye(18) - gain @ H).T + gain @ R @ gain.T
        kf.update(offered)
        np.testing.assert_allclose(kf.P, joseph, rtol=1e-6, atol=1e-11)
