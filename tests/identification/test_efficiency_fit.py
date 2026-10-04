from pathlib import Path

import numpy as np

from virtualmodelcontrol.identification import fit_efficiency, plateaus
from virtualmodelcontrol.models import Efficiency

DATA = np.load(Path(__file__).parents[1] / "data" / "efficiency.npz")
rng = np.random.default_rng(3)


def test_plateaus_match_the_lab_detection():
    windows = plateaus(DATA["log_t"], DATA["log_dref"], 2e-3, min_hold=8.0, steady_fraction=0.5)
    i0, i1 = DATA["plateau_i0"], DATA["plateau_i1"]
    assert [w.stop for w in windows] == i1.tolist()
    assert [w.start for w in windows] == (i0 + ((i1 - i0) * 0.5).astype(int)).tolist()
    for name in ("force", "spring", "struct"):
        means = [DATA[f"log_{name}"][w].mean() for w in windows]
        np.testing.assert_allclose(means, DATA[f"plateau_{name}"], rtol=1e-12)
    stable = [DATA["log_force"][w].std() < 0.03 for w in windows]
    assert stable == DATA["plateau_stable"].tolist()


def test_the_lab_fit_through_the_origin_on_static_points():
    keep = DATA["plateau_stable"] & (np.abs(DATA["plateau_force"]) >= 0.05)
    commanded = DATA["plateau_spring"][keep]  # the virtual spring's force along the normal
    delivered = (DATA["plateau_force"] + DATA["plateau_struct"])[keep]
    efficiency = fit_efficiency(commanded, delivered)
    np.testing.assert_allclose(efficiency.params["c1"].value, [DATA["arms_eta"]], rtol=1e-12)


def test_the_finger_motor_efficiencies_from_its_recorded_data():
    efficiency = fit_efficiency(
        DATA["finger_commanded"], DATA["finger_measured"], weights=DATA["finger_weights"]
    )
    np.testing.assert_allclose(efficiency.params["c1"].value, DATA["finger_eta"], rtol=1e-10)
    assert efficiency.degree == 1


def test_polynomials_are_recovered_motor_by_motor_and_shared():
    truth = Efficiency((0.9, 0.5), (-0.4, 0.2), (1.5, -0.8))
    u = rng.uniform(-0.5, 0.5, (40, 2))
    delivered = truth(u)
    fitted = fit_efficiency(u, delivered, degree=3)
    for k in ("c1", "c2", "c3"):
        np.testing.assert_allclose(fitted.params[k].value, truth.params[k].value, atol=1e-12)
    w = rng.uniform(0.5, 2.0, (40, 2))  # one measurement per point: Σ w_j τ_j
    fitted = fit_efficiency(u, (w * delivered).sum(axis=1), degree=3, weights=w)
    for k in ("c1", "c2", "c3"):
        np.testing.assert_allclose(fitted.params[k].value, truth.params[k].value, atol=1e-10)
    common = Efficiency(0.7, -0.3)
    fitted = fit_efficiency(u, common(u), degree=2, shared=True)
    assert fitted.params["c1"].shape == ()
    np.testing.assert_allclose([fitted.params["c1"].value, fitted.params["c2"].value], [0.7, -0.3])
    fitted = fit_efficiency(u, (w * common(u)).sum(axis=1), degree=2, weights=w, shared=True)
    np.testing.assert_allclose([fitted.params["c1"].value, fitted.params["c2"].value], [0.7, -0.3])


def test_plateaus_hold_any_number_of_columns():
    t = np.arange(0.0, 30.0, 0.5)
    held = np.zeros((t.size, 2))
    held[t >= 10.0, 1] = 1.0  # one column steps at 10 s
    assert [(w.start, w.stop) for w in plateaus(t, held, 0.5)] == [(10, 20), (40, 60)]
    assert plateaus(t, held, 0.5, min_hold=20.0) == []
    assert [(w.start, w.stop) for w in plateaus(t, held, 0.5, steady_fraction=1.0)] == [
        (0, 20),
        (20, 60),
    ]
