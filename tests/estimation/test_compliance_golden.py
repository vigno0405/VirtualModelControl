"""Object compliance against the lab's notebook, run on made-up probes of the hand's experiment.

The fixture holds three runs of a probe at a gentle setting (the baseline) and at three stiffer
ones, with outliers among the samples, and the lab's compliance [m/N] at each stiffer setting.
"""

from pathlib import Path

import numpy as np
import pytest

from virtualmodelcontrol.estimation import object_compliance

DATA = np.load(Path(__file__).parents[1] / "data" / "compliance.npz")
GENTLE, SWEEP = int(DATA["k_gentle"]), [int(k) for k in DATA["k_sweep"]]


def probe(run, k):
    return DATA[f"run{run}_k{k}_position"], DATA[f"run{run}_k{k}_force"]


def test_the_compliance_of_every_probe_is_the_labs():
    for run in range(int(DATA["n_runs"])):
        for k in SWEEP:
            got = object_compliance(*probe(run, k), *probe(run, GENTLE))
            assert got == pytest.approx(DATA[f"lab_k{k}"][run], rel=1e-12)


def test_a_probe_of_a_spring_gives_back_its_compliance():
    rng = np.random.default_rng(2)
    compliance = 4.2e-3  # [m/N]
    x0, f0 = rng.normal(0, 0.02, 3), rng.normal(0, 0.3, 3)
    push = np.array([0.0, 0.6, -0.8]) * 2.5  # [N]
    got = object_compliance(x0 + compliance * push, f0 + push, x0, f0)
    assert got == pytest.approx(compliance, rel=1e-12)  # also from single samples, in 3-vectors


def test_a_probe_that_did_not_change_the_force_has_no_compliance():
    x, f = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    assert np.isnan(object_compliance(x + 0.01, f, x, f))


def test_the_median_ignores_a_few_wild_samples():
    position, force = probe(0, SWEEP[0])
    clean = object_compliance(position, force, *probe(0, GENTLE))
    wild = position.copy()
    wild[:2] += 1.0  # [m]: two samples of forty are wrong
    assert object_compliance(wild, force, *probe(0, GENTLE)) == pytest.approx(clean, rel=0.05)


def test_missing_samples_do_not_count():
    position, force = probe(1, SWEEP[1])
    clean = object_compliance(position, force, *probe(1, GENTLE))
    holes = position.copy()
    holes[::5] = np.nan  # a marker lost now and then
    assert object_compliance(holes, force, *probe(1, GENTLE)) == pytest.approx(clean, rel=0.05)


def test_a_force_change_of_a_tenth_of_a_piconewton_is_none():
    x, f = np.zeros(3), np.array([0.0, 0.0, 1.0])
    up = np.array([0.0, 0.0, 1.0])
    assert np.isnan(object_compliance(x + 1e-3 * up, f + 1e-13 * up, x, f))
    assert object_compliance(x + 1e-3 * up, f + 1e-3 * up, x, f) == pytest.approx(1.0)
