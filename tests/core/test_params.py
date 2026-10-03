import casadi as ca
import numpy as np
import pytest

from virtualmodelcontrol.core import Binding, Param, ParamSet, as_param, constants


def test_param_keeps_its_shape():
    p = Param("delta", [0.0, 2.1, -2.1], unit="rad", scope="design")
    p.value = np.array([[1.0], [2.0], [3.0]])  # same size, reshaped
    assert p.shape == (3,)
    np.testing.assert_array_equal(p.value, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="shape"):
        p.value = [1.0, 2.0]


def test_param_rejects_unknown_scope_and_tensors():
    with pytest.raises(ValueError, match="scope"):
        Param("k", 1.0, scope="sometimes")
    with pytest.raises(ValueError, match="scalar, vector or matrix"):
        Param("t", np.zeros((2, 2, 2)))


def test_as_param_passes_params_through():
    p = Param("k", 3.0)
    assert as_param(p, "other") is p
    q = as_param(4.0, "c", unit="N*s/m", scope="stage")
    assert (q.name, float(q.value), q.unit, q.scope) == ("c", 4.0, "N*s/m", "stage")


def test_paramset_names_and_dedup_by_identity():
    a, b = Param("k", 1.0), Param("k", 2.0)
    ps = ParamSet([a])
    assert ps.add(a, "other") == "k"  # same object keeps its first name
    with pytest.raises(ValueError, match="duplicate"):
        ps.add(b)
    assert ps.add(b, rename=True) == "k2"
    assert ps.name_of(b) == "k2" and ps.has(b) and len(ps) == 2


def test_paramset_merge_and_select():
    inner = ParamSet([Param("stiffness", 1.0, scope="stage"), Param("s", 0.5, scope="episode")])
    outer = ParamSet()
    outer.merge(inner, "drag")
    assert list(outer) == ["drag.stiffness", "drag.s"]
    assert outer.select(scopes=["stage"]) == ["drag.stiffness"]
    assert outer.select(patterns=["*.s"]) == ["drag.s"]


def test_vector_round_trip_packs_matrices_by_column():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    ps = ParamSet([Param("a", 5.0), Param("v", [6.0, 7.0]), Param("m", m)])
    x = ps.vector()
    np.testing.assert_array_equal(x, [5.0, 6.0, 7.0, 1.0, 3.0, 2.0, 4.0])
    ps.set_vector(x * 2)
    np.testing.assert_array_equal(ps["m"].value, 2 * m)
    with pytest.raises(ValueError, match="expected 7"):
        ps.set_vector(np.zeros(3))


def test_dict_round_trip():
    ps = ParamSet([Param("a", 1.0), Param("v", [2.0, 3.0])])
    d = ps.to_dict()
    ps.update({"a": 9.0})
    assert float(ps["a"].value) == 9.0
    ps.update(d)
    assert ps.to_dict() == {"a": 1.0, "v": [2.0, 3.0]}


def test_binding_live_slices_and_frozen_constants():
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    k, v, mat = Param("k", 2.0, scope="stage"), Param("v", [1.0, -1.0]), Param("m", m)
    ps = ParamSet([k, v, mat])
    b = Binding(ps, live=["k", "m"])
    assert b.live == ["k", "m"] and b.p.numel() == 5
    assert isinstance(b(v), ca.DM)  # frozen: a constant
    expr = b(k) * ca.mtimes(b(mat), b(v))
    f = ca.Function("f", [b.p], [expr])
    np.testing.assert_allclose(np.array(f(b.values())).ravel(), 2.0 * m @ [1.0, -1.0])
    view = b.view(ps)
    assert set(view) == {"k", "v", "m"}
    with pytest.raises(KeyError, match="unknown live"):
        Binding(ps, live=["nope"])


def test_constants_view():
    ps = ParamSet([Param("v", [1.0, 2.0])])
    np.testing.assert_array_equal(np.array(constants(ps)["v"]).ravel(), [1.0, 2.0])
