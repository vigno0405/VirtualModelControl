from itertools import pairwise

import numpy as np

from virtualmodelcontrol.estimation import VelocityFilter

DT = 0.01
SLOPE = np.array([2.0, -3.0, 0.5])


def ramp(k):
    """The sample k of a ramp of slope SLOPE, one sample per DT."""
    return SLOPE * k * DT


def test_a_ramp_gives_its_slope_once_the_filter_has_settled():
    vf = VelocityFilter(DT, alpha=0.3)
    for k in range(40):
        v = vf.update(ramp(k))
    np.testing.assert_allclose(v, SLOPE, rtol=1e-12)


def test_the_low_pass_is_first_order_and_the_default_keeps_three_tenths_of_the_last_velocity():
    vf = VelocityFilter(DT)  # alpha = 0.3
    for k in range(8):
        np.testing.assert_allclose(vf.update(ramp(k)), (1.0 - 0.3**k) * SLOPE, rtol=1e-12)
    slow = VelocityFilter(DT, alpha=0.8)
    for k in range(8):
        np.testing.assert_allclose(slow.update(ramp(k)), (1.0 - 0.8**k) * SLOPE, rtol=1e-12)


def test_the_first_sample_gives_zeros_and_so_does_the_first_after_a_reset():
    vf = VelocityFilter(DT)
    np.testing.assert_array_equal(vf.update([1.0, 2.0]), [0.0, 0.0])
    assert np.abs(vf.update([1.1, 2.0])).max() > 1.0
    vf.reset()
    np.testing.assert_array_equal(vf.update([5.0, 5.0]), [0.0, 0.0])  # no jump from before it
    np.testing.assert_allclose(vf.update([5.1, 5.0]), [0.7 * 0.1 / DT, 0.0], rtol=1e-12)


def test_alpha_zero_is_the_raw_finite_difference_over_dt():
    rng = np.random.default_rng(3)
    samples = rng.normal(size=(12, 4))
    vf = VelocityFilter(0.25, alpha=0.0)
    vf.update(samples[0])
    for last, now in pairwise(samples):
        np.testing.assert_allclose(vf.update(now), (now - last) / 0.25, rtol=1e-14)


def test_the_filter_takes_a_vector_of_any_size():
    for size in (1, 2, 5, 12):
        vf = VelocityFilter(DT, alpha=0.0)
        assert vf.update(np.zeros(size)).shape == (size,)
        np.testing.assert_allclose(vf.update(np.full(size, DT)), np.ones(size), rtol=1e-12)
    vf = VelocityFilter(DT)
    assert vf.update([1, 2, 3]).dtype == float  # integers in, floats out


def test_the_velocity_and_the_samples_are_copies_that_the_caller_may_change():
    vf = VelocityFilter(DT, alpha=0.5)
    sample = np.array([1.0, 1.0])
    vf.update(sample)
    sample[:] = 100.0  # the filter keeps its own copy of the last sample
    v = vf.update([1.0 + DT, 1.0])
    np.testing.assert_allclose(v, [0.5, 0.0], rtol=1e-12)
    v[:] = 100.0  # and of the velocity it will low-pass
    np.testing.assert_allclose(vf.update([1.0 + 2 * DT, 1.0]), [0.75, 0.0], rtol=1e-12)
