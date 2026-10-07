"""The models as numpy and PyTorch code: the same numbers as CasADi, and nothing but numpy."""

import subprocess
import sys

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from backend_helpers import arm, check, flywheel, operations, rng, same
from virtualmodelcontrol.core import backends
from virtualmodelcontrol.robots import turtle


def test_every_operation_has_a_translation_equal_to_casadis():
    function, names = operations()
    assert len(names) > 40
    for x in [*(rng.uniform(0.2, 0.9, 4) for _ in range(5)), np.full(4, 0.5), [0.3, 0.3, 0.7, 0.7]]:
        check(function, [np.asarray(x)])  # the last two are ties: <= is not <, == is not !=
    codes = {function.expand().instruction_id(k) for k in range(function.expand().n_instructions())}
    assert {
        getattr(ca, f"OP_{op}") for op in ("FMOD", "COPYSIGN", "ATAN2", "IF_ELSE_ZERO", "HYPOT")
    } <= codes


def test_shapes_sparsity_constants_and_batches_come_out_as_casadis_do():
    x, M = ca.SX.sym("x", 3), ca.SX.sym("M", 2, 3)
    y = ca.SX.sym("y")
    sparse = ca.jacobian(ca.vertcat(x[0] * x[1], x[2] ** 2, 5.0), x)  # structural zeros
    function = ca.Function(
        "shapes",
        [x, M, y],
        [ca.mtimes(M, x), sparse, ca.SX(2, 2), ca.DM([[1.0, 2.0], [3.0, 4.0]]), y * y, M.T],
    )
    for _ in range(3):
        args = [rng.normal(size=3), rng.normal(size=(2, 3)), rng.normal()]
        run = check(function, args)
        shapes = [np.shape(o) for o in run(*args)]
        assert shapes == [(2,), (3, 3), (2, 2), (2, 2), (), (3, 2)]
    assert run.source.count("def ") >= 1 and "import casadi" not in run.source
    batch = [rng.normal(size=(5, 3)), rng.normal(size=(5, 2, 3)), rng.normal(size=5)]
    got = run(*batch)
    for i in range(5):
        for g, w in zip(got, function(batch[0][i], batch[1][i], batch[2][i]), strict=True):
            same(np.asarray(g)[i], w)
    shared = run(rng.normal(size=3), rng.normal(size=(2, 3)), rng.normal(size=4))
    assert shared[0].shape == (4, 2) and shared[4].shape == (4,)  # every output has the batch


def test_infinities_and_not_a_number_are_written_so_that_python_reads_them():
    x = ca.SX.sym("x")
    function = ca.Function(
        "odd", [x], [ca.vertcat(x + ca.inf, x + ca.SX(-ca.inf), x * 0 + ca.SX(np.nan))]
    )
    got = backends.translate(function)(1.0)
    assert got[0] == np.inf and got[1] == -np.inf and np.isnan(got[2])


def test_export_keeps_the_functions_that_are_missing_and_the_casadi_ones_as_they_are():
    x = ca.SX.sym("x")
    function = ca.Function("sq", [x], [x * x])
    assert backends.export({"a": function}, "casadi").a is function
    out = backends.export({"a": None, "sq": function}, "numpy")
    assert out.a is None and out.sq(3.0) == 9.0 and "def sq" in out.sq.source


def test_a_function_with_no_inputs_and_an_unknown_operation_are_handled():
    constant = ca.Function("constant", [], [ca.DM([1.0, 2.0]) * 3])
    np.testing.assert_allclose(backends.translate(constant)(), [3.0, 6.0])
    x = ca.SX.sym("x")
    with pytest.raises(NotImplementedError, match="ERFINV"):
        backends.source(ca.Function("bad", [x], [ca.erfinv(x)]))
    with pytest.raises(ValueError, match="backend"):
        backends.source(constant, "jax")
    mx = ca.MX.sym("m", 2)
    assert backends.translate(ca.Function("mx", [mx], [ca.sin(mx) * 2]))([0.5, 1.0]).shape == (2,)


def test_the_kinematics_of_a_soft_arm_agree_to_rounding_error_with_their_jacobians_and_hessians():
    kin = vmc.Kinematics(arm())
    for site in (1.0, 0.5):
        function = kin.functions(site)
        run = kin.functions(site, backend="numpy")
        assert kin.functions(site, backend="numpy") is run  # built once
        for _ in range(3):
            q = rng.normal(size=9) * 0.004
            for g, w in zip(run(q), function(q), strict=True):
                same(g, w)
    assert run.source.startswith('"""') and "casadi" not in run.source.split("def ")[1]


def test_the_dynamics_of_a_crawler_with_its_contacts_and_a_floating_body_agree():
    body = turtle.crawler()
    dynamics = vmc.compile_dynamics(body)
    out = dynamics.export("numpy")
    assert out.forward.source and callable(out.step) and out.elements is not None
    q = np.array([0.1, -0.2, 0.01, 1, 0, 0, 0, 0.3, -2.0])
    q[3:7] = vmc.Quaternion().integrate(q[3:7], [0.1, -0.2, 0.3])
    v, a = rng.normal(size=8) * 0.3, rng.normal(size=8)
    u, p, t = rng.normal(size=2), dynamics.live_values(), 0.37
    for name, args in {
        "forward": (q, v, u, p, t),
        "residual": (q, v, a, u, p, t),
        "mass": (q, p),
        "energy": (q, v, p, t),
        "power": (q, v, u, p, t),
        "motors": (q, v, p),
        "step": (q, v, u, p, t, 0.002),
        "elements": (q, v, p, t),
    }.items():
        function = getattr(dynamics, name)
        got = getattr(out, name)(*args)
        want = function(*args)
        got, want = (got, want) if function.n_out() > 1 else ((got,), (want,))
        for g, w in zip(got, want, strict=True):
            same(g, w)


def test_a_compiled_controller_with_a_virtual_state_agrees_in_every_function():
    compiled = vmc.compile(flywheel())
    out = compiled.export("numpy")
    q, v, z = rng.normal(size=2), rng.normal(size=2), rng.normal(size=2)
    p, t = compiled.live_values(), 0.8
    motors = np.concatenate([q, v, z, p, [t]])
    for name in ("law", "tau", "energy", "power", "forces"):
        function = getattr(compiled, name)
        got = getattr(out, name)(q, v, z, p, t)
        want = function(q, v, z, p, t)
        got, want = (got, want) if function.n_out() > 1 else ((got,), (want,))
        for g, w in zip(got, want, strict=True):
            same(g, w)
    for name in ("fast", "fast_energy", "fast_power", "fast_elements"):
        function = getattr(compiled, name)
        got, want = getattr(out, name)(motors), function(motors)
        got, want = (got, want) if function.n_out() > 1 else ((got,), (want,))
        for g, w in zip(got, want, strict=True):
            same(g, w)


def test_the_source_runs_with_numpy_alone(tmp_path):
    dynamics = vmc.compile_dynamics(arm())
    out = dynamics.export("numpy")
    q, v, u = rng.normal(size=9) * 0.004, rng.normal(size=9) * 0.01, rng.normal(size=9)
    p = dynamics.live_values()
    path = tmp_path / "forward.py"
    path.write_text(out.forward.source, encoding="utf-8")
    want = np.array(dynamics.forward(q, v, u, p, 0.0)).ravel()
    program = (
        "import sys\n"
        "sys.modules['casadi'] = None\n"  # importing casadi now raises
        "import importlib.util, numpy as np\n"
        f"spec = importlib.util.spec_from_file_location('forward', {str(path)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        f"q, v, u, p = np.load({str(tmp_path / 'in.npz')!r}).values()\n"
        "print(repr(module.forward(q, v, u, p, 0.0).tolist()))\n"
    )
    np.savez(tmp_path / "in.npz", q=q, v=v, u=u, p=p)
    run = subprocess.run(
        [sys.executable, "-I", "-c", program], capture_output=True, text=True, timeout=120
    )
    assert run.returncode == 0, run.stderr
    got = np.array(eval(run.stdout))
    same(got, want)
    assert "casadi" not in out.forward.source
