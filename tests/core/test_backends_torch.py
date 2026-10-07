"""The models as PyTorch code: CasADi's numbers, and derivatives by autograd."""

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from backend_helpers import arm, check, flywheel, operations, rng, same
from virtualmodelcontrol.core import backends

torch = pytest.importorskip("torch")


def test_torch_gives_casadis_numbers_in_float64_with_a_batch_and_follows_a_device_dtype():
    dynamics = vmc.compile_dynamics(arm())
    out = dynamics.export("torch")
    q = torch.as_tensor(rng.normal(size=(6, 9)) * 0.004)
    v, u = (
        torch.as_tensor(rng.normal(size=(6, 9)) * 0.01),
        torch.as_tensor(rng.normal(size=(6, 9)) * 0.05),
    )
    p = torch.as_tensor(dynamics.live_values())
    acc = out.forward(q, v, u, p, 0.0)
    assert acc.dtype == torch.float64 and acc.shape == (6, 9)
    for i in range(6):
        want = dynamics.forward(q[i].numpy(), v[i].numpy(), u[i].numpy(), p.numpy(), 0.0)
        # PyTorch's sin and cos differ from libm in the last place, and the mass matrix of a soft
        # arm has a condition number of a few million: a few parts in 1e11
        same(acc[i].numpy(), want, 1e-9)
    single = out.forward(q[0].float(), v[0].float(), u[0].float(), p.float(), 0.0)
    assert single.dtype == torch.float32  # the code takes the dtype of its arguments


def test_torch_differentiates_a_model_as_casadi_does():
    kin = vmc.Kinematics(arm())
    function, run = kin.functions(1.0), kin.functions(1.0, backend="torch")
    q = rng.normal(size=9) * 0.004
    # the position's Jacobian by autograd is the J that CasADi's own derivative gives
    jac = torch.autograd.functional.jacobian(lambda x: run(x)[0], torch.as_tensor(q))
    same(jac.numpy(), function(q)[2])
    # and the Hessian of one coordinate of the position is its slice of H
    hess = torch.autograd.functional.hessian(lambda x: run(x)[0][2], torch.as_tensor(q))
    same(hess.numpy(), np.array(function(q)[4])[18:27])
    dynamics = vmc.compile_dynamics(arm())
    out = dynamics.export("torch")
    p = torch.as_tensor(dynamics.live_values())
    v, u = torch.zeros(9, dtype=torch.float64), torch.zeros(9, dtype=torch.float64)
    grad = torch.autograd.functional.jacobian(
        lambda x: out.forward(x, v, u, p, 0.0), torch.as_tensor(q)
    )
    jacobian = dynamics.forward.jacobian()  # of the acceleration, input by input: q first
    want = jacobian(q, v.numpy(), u.numpy(), p.numpy(), 0.0, np.zeros(9))[0]
    same(grad.numpy(), want, 1e-8)
    x = torch.as_tensor(q).clone().requires_grad_(True)
    (out.energy(x, v, p, 0.0)[1]).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().max() > 0


def test_torch_runs_every_operation_and_the_wrappers_take_the_backend_by_name():
    function, _ = operations()
    for x in [*(rng.uniform(0.2, 0.9, 4) for _ in range(3)), np.full(4, 0.5), [0.3, 0.3, 0.7, 0.7]]:
        check(function, [np.asarray(x)], "torch")
    x = ca.SX.sym("x", 2)
    square = backends.translate(ca.Function("square", [x], [x * x]), "torch")
    assert square(np.array([2, 3])).dtype == torch.float64  # whole numbers are made floats
    assert square(torch.tensor([2, 3])).dtype == torch.float64
    x, M, y = ca.SX.sym("x", 3), ca.SX.sym("M", 2, 3), ca.SX.sym("y")
    shapes = ca.Function("s", [x, M, y], [ca.mtimes(M, x), ca.jacobian(x[0] * x[1], x), y * y, M.T])
    check(shapes, [rng.normal(size=3), rng.normal(size=(2, 3)), rng.normal()], "torch")
    compiled = vmc.compile(flywheel())
    out = compiled.export("torch")
    q, v, z = (torch.as_tensor(rng.normal(size=2)) for _ in range(3))
    p = torch.as_tensor(compiled.live_values())
    u, zdot = out.law(q, v, z, p, 0.8)
    want = compiled.law(q.numpy(), v.numpy(), z.numpy(), p.numpy(), 0.8)
    same(u.numpy(), want[0])
    same(zdot.numpy(), want[1])
    assert backends.export({"a": None}, "casadi").a is None
