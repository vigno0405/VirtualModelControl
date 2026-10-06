import casadi as ca
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

import virtualmodelcontrol as vmc

m = vmc.math
rng = np.random.default_rng(3)


def turn(size, limit=3.0):
    """A random vector whose first three entries (the rotation) have an angle below ``limit``."""
    x = rng.normal(size=size)
    return x * min(1.0, limit / np.linalg.norm(x[:3]))


VECTORS = [turn(3) for _ in range(5)] + [np.zeros(3), np.array([1e-9, -2e-9, 0.0])]
TWISTS = [turn(6) for _ in range(5)] + [
    np.zeros(6),
    np.array([1e-9, 0, 2e-9, 0.1, 0.2, 0.3]),
    np.array([2e-3, -1e-3, 3e-3, 0.4, -0.5, 0.6]),  # in the range of the series
]


def rotation_of(w):
    return Rotation.from_rotvec(w).as_matrix()


def symbolic(fn, *shapes):
    """``fn`` run on CasADi symbols and compiled, so that it can be called on numbers."""
    args = [ca.SX.sym(f"a{i}", *shape) for i, shape in enumerate(shapes)]
    return ca.Function("f", args, [fn(*args)])


@pytest.mark.parametrize("axis", "xyz")
def test_the_axis_rotations_agree_with_scipy(axis):
    angle = 0.7
    expected = Rotation.from_euler(axis, angle).as_matrix()
    np.testing.assert_allclose(getattr(m, f"rot_{axis}")(angle), expected, atol=1e-15)
    unit = np.eye(3)["xyz".index(axis)]
    np.testing.assert_allclose(m.rot(unit, angle), expected, atol=1e-15)


@pytest.mark.parametrize("w", VECTORS)
def test_exp_so3_is_the_rotation_of_the_vector(w):
    R = m.exp_so3(w)
    np.testing.assert_allclose(R, rotation_of(w), atol=1e-12)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("w", VECTORS)
def test_log_so3_undoes_exp_so3(w):
    np.testing.assert_allclose(m.log_so3(m.exp_so3(w)), w, atol=1e-8)


def test_skew_and_vee():
    w, x = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(m.skew(w) @ x, np.cross(w, x), atol=1e-15)
    np.testing.assert_allclose(m.vee(m.skew(w)), w)


@pytest.mark.parametrize("twist", TWISTS)
def test_exp_se3_is_the_matrix_exponential_of_the_twist(twist):
    hat = np.zeros((4, 4))
    hat[:3, :3], hat[:3, 3] = m.skew(twist[:3]), twist[3:]
    np.testing.assert_allclose(m.exp_se3(twist), expm(hat), atol=1e-12)


@pytest.mark.parametrize("twist", TWISTS)
def test_log_se3_undoes_exp_se3(twist):
    np.testing.assert_allclose(m.log_se3(m.exp_se3(twist)), twist, atol=1e-8)


def test_the_inverse_and_the_adjoint_of_a_transform():
    T = m.exp_se3(turn(6))
    np.testing.assert_allclose(m.invert(T) @ T, np.eye(4), atol=1e-12)
    xi = rng.normal(size=6)
    moved = T @ m.exp_se3(xi) @ m.invert(T)
    np.testing.assert_allclose(moved, m.exp_se3(m.adjoint(T) @ xi), atol=1e-12)
    np.testing.assert_allclose(m.transform(T[:3, :3], T[:3, 3]), T, atol=1e-15)


CASES = {
    "rot_x": (m.rot_x, [()]),
    "rot_y": (m.rot_y, [()]),
    "rot_z": (m.rot_z, [()]),
    "skew": (m.skew, [(3,)]),
    "exp_so3": (m.exp_so3, [(3,)]),
    "log_so3": (lambda R: m.log_so3(R), [(3, 3)]),
    "exp_se3": (m.exp_se3, [(6,)]),
    "log_se3": (m.log_se3, [(4, 4)]),
    "invert": (m.invert, [(4, 4)]),
    "adjoint": (m.adjoint, [(4, 4)]),
    "rot": (m.rot, [(3,), ()]),
    "transform": (m.transform, [(3, 3), (3,)]),
}


def sample(shape):
    if shape == (3, 3):
        return rotation_of(turn(3))
    if shape == (4, 4):
        return m.exp_se3(turn(6))
    return rng.normal(size=shape) if shape else float(rng.normal())


@pytest.mark.parametrize("name", CASES)
def test_numbers_and_symbols_give_the_same_values(name):
    fn, shapes = CASES[name]
    values = [sample(s) for s in shapes]
    out = fn(*values)
    assert isinstance(out, np.ndarray)
    sym = symbolic(fn, *[s if s else (1, 1) for s in shapes])
    np.testing.assert_allclose(np.array(sym(*values)).reshape(np.shape(out)), out, atol=1e-12)


def hessian_at(fn, x0, n_out):
    """The Hessian of each output of ``fn`` at ``x0``, as the difference of its Jacobians."""
    x = ca.SX.sym("x", len(x0))
    jac = ca.Function("j", [x], [ca.jacobian(ca.vec(fn(x)), x)])
    h, n = 1e-5, len(x0)
    out = np.zeros((n_out, n, n))
    for i in range(n):
        step = np.zeros(n)
        step[i] = h
        out[:, :, i] = (np.array(jac(x0 + step)) - np.array(jac(x0 - step))) / (2 * h)
    ana = ca.Function(
        "h", [x], [ca.vertcat(*[ca.hessian(ca.vec(fn(x))[k], x)[0] for k in range(n_out)])]
    )
    return np.array(ana(x0)).reshape(n_out, n, n), out


@pytest.mark.parametrize(
    ("name", "fn", "n_in", "n_out"),
    [
        ("exp_so3", m.exp_so3, 3, 9),
        ("exp_se3", m.exp_se3, 6, 16),
        ("log_so3", lambda w: m.log_so3(m.exp_so3(w)), 3, 3),
        ("log_se3", lambda xi: m.log_se3(m.exp_se3(xi)), 6, 6),
    ],
)
@pytest.mark.parametrize("where", ["zero", "small", "generic"])
def test_second_derivatives_are_right_also_at_the_neutral_point(name, fn, n_in, n_out, where):
    x0 = {
        "zero": np.zeros(n_in),
        "small": 1e-4 * np.arange(1, n_in + 1),
        "generic": 0.3 * rng.normal(size=n_in),
    }[where]
    exact, differenced = hessian_at(fn, x0, n_out)
    np.testing.assert_allclose(exact, differenced, atol=1e-5)
