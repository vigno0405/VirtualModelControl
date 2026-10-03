import casadi as ca
import numpy as np
import pytest

from helpers import Rod
from virtualmodelcontrol.core import Binding, ParamSet
from virtualmodelcontrol.mechanisms import (
    Context,
    Custom,
    Difference,
    FramePoint,
    Joint,
    Norm,
    Projection,
    Ref,
    Stack,
    State,
    walk,
)

X = Joint(slice(0, 3), unit="m")


def value(coord, q, params=None, z=None, states=None):
    ps = params if params is not None else ParamSet()
    for c in walk(coord):
        for name, p in c.params().items():
            ps.add(p, name, rename=True)
    ctx = Context(ca.DM(q), Binding(ps), z=None if z is None else ca.DM(z), states=states)
    return np.array(ca.DM(ctx.value(coord))).ravel()


def test_subtracting_an_array_makes_a_live_reference():
    d = X - [1.0, 2.0, 3.0]
    assert isinstance(d, Difference) and isinstance(d.b, Ref)
    assert d.b.param.scope == "stage" and d.b.param.unit == "m"
    np.testing.assert_allclose(value(d, [1.0, 1.0, 1.0]), [0.0, -1.0, -2.0])
    np.testing.assert_allclose(value([1.0, 2.0, 3.0] - X, [1.0, 1.0, 1.0]), [0.0, 1.0, 2.0])


def test_slice_stack_projection_norm():
    q = [3.0, 4.0, 12.0]
    np.testing.assert_allclose(value(X[1:], q), [4.0, 12.0])
    np.testing.assert_allclose(value(Stack(X[0], X[2]), q), [3.0, 12.0])
    np.testing.assert_allclose(value(Projection(X, [0.0, 0.0, 2.0]), q), [12.0])  # n normalized
    np.testing.assert_allclose(value(Norm(X), q), [13.0])


def test_norm_has_a_finite_gradient_at_zero():
    q = ca.SX.sym("q", 3)
    ctx = Context(q, Binding(ParamSet()))
    g = ca.Function("g", [q], [ca.jacobian(ctx.value(Norm(X)), q)])
    assert np.all(np.isfinite(np.array(g([0.0, 0.0, 0.0]))))


def test_custom_coordinate_with_params():
    c = Custom(lambda x, k: k * x[0] ** 2, [X], dim=1, params={"k": 2.0})
    assert set(c.params()) == {"k"}
    np.testing.assert_allclose(value(c, [3.0, 0.0, 0.0]), [18.0])


def test_walk_visits_shared_children_once():
    shared = X[0]
    nodes = list(walk(Stack(shared, shared, X)))
    assert len(nodes) == 3 and len({id(n) for n in nodes}) == 3


def test_joint_indices():
    assert Joint([2, 0]).indices == [2, 0]
    np.testing.assert_allclose(value(Joint([2, 0]), [1.0, 2.0, 3.0]), [3.0, 1.0])
    with pytest.raises(ValueError, match="explicit stop"):
        Joint(slice(1, None))


def test_frame_point_on_a_rod():
    rod = Rod(length=2.0)
    p = FramePoint(rod, s=0.25, offset=[0.1, 0.0, 0.0])
    assert p.s.scope == "episode" and p.s.bounds == (0.0, 1.0)
    params = ParamSet()
    params.merge(rod.params)
    np.testing.assert_allclose(value(p, [0.0, 0.0, 1.0], params), [0.1, 0.0, 1.5])
    params = ParamSet()
    params.merge(rod.params)
    np.testing.assert_allclose(value(FramePoint(rod, "tip"), np.zeros(3), params), [0, 0, 2.0])
    with pytest.raises(ValueError, match="site name"):
        FramePoint(rod)


def test_state_reads_its_slice_of_z():
    a, b = State("a", 1), State("b", 2, initial=[1.0, 2.0])
    np.testing.assert_array_equal(b.initial, [1.0, 2.0])
    states = {id(a): slice(0, 1), id(b): slice(1, 3)}
    np.testing.assert_allclose(value(b, np.zeros(3), z=[5.0, 6.0, 7.0], states=states), [6, 7])
