"""Scaled decision variables."""

import casadi as ca
import numpy as np
import pytest

from virtualmodelcontrol.optimization.variables import Variables


def test_variables_stack_in_order_with_scaled_bounds_and_starts():
    v = Variables()
    a = v.add("a", 2, -1.0, [1.0, 2.0], [0.5, 1.0], scale=10.0)
    b = v.add("b", 1, 0.0, np.inf, 3.0)
    assert v.size == 3
    assert v.slices == {"a": slice(0, 2), "b": slice(2, 3)}
    np.testing.assert_allclose(v.lower, [-0.1, -0.1, 0.0])
    np.testing.assert_allclose(v.upper, [0.1, 0.2, np.inf])
    np.testing.assert_allclose(v.init, [0.05, 0.1, 3.0])
    # the model sees the physical values: the solver's symbols times their scales
    physical = ca.Function("physical", [v.x], [ca.vertcat(a, b)])
    np.testing.assert_allclose(np.array(physical([1.0, 2.0, 3.0])).ravel(), [10.0, 20.0, 3.0])


def test_physical_and_scaled_values_convert_both_ways():
    v = Variables()
    v.add("a", 2, -np.inf, np.inf, 0.0, scale=4.0)
    v.add("b", 1, -np.inf, np.inf, 0.0)
    x = np.array([1.0, 2.0, 5.0])
    np.testing.assert_allclose(v.value(x, "a"), [4.0, 8.0])
    np.testing.assert_allclose(v.scaled("a", [4.0, 8.0]), [1.0, 2.0])
    np.testing.assert_allclose(v.value(x, "b"), [5.0])


def test_a_name_is_used_once():
    v = Variables()
    v.add("a", 1, 0.0, 1.0, 0.5)
    with pytest.raises(ValueError, match="already exists"):
        v.add("a", 1, 0.0, 1.0, 0.5)


def test_an_empty_set_has_empty_vectors():
    v = Variables()
    assert v.size == 0 and v.lower.size == v.upper.size == v.init.size == 0
