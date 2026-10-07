"""A CasADi function as plain Python: source for numpy or PyTorch, generated from its graph.

The models of the library are CasADi functions. ``source`` writes one as a straight-line Python
function over numpy arrays or PyTorch tensors, one statement per operation of its expression
graph, and ``translate`` runs that source. The text imports numpy (or torch) and nothing else, so
a model can leave CasADi behind: a controller on a machine without it, a network that is trained
through a model. The operations are the same ones in the same order, so the values agree with
CasADi to rounding error, and PyTorch differentiates the model like any other function.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

import casadi as ca

BACKENDS = ("numpy", "torch")

UNARY = {
    "EXP": "exp",
    "LOG": "log",
    "SQRT": "sqrt",
    "SIN": "sin",
    "COS": "cos",
    "TAN": "tan",
    "ASIN": "arcsin",
    "ACOS": "arccos",
    "ATAN": "arctan",
    "SINH": "sinh",
    "COSH": "cosh",
    "TANH": "tanh",
    "ASINH": "arcsinh",
    "ACOSH": "arccosh",
    "ATANH": "arctanh",
    "LOG1P": "log1p",
    "EXPM1": "expm1",
    "FLOOR": "floor",
    "CEIL": "ceil",
    "FABS": "abs",
    "SIGN": "sign",
    "ERF": "erf",
}
"""Operations of one argument, by CasADi's name, and the numpy name of the function."""

TORCH_NAMES = {"arcsin": "asin", "arccos": "acos", "arctan": "atan", "arcsinh": "asinh",
               "arccosh": "acosh", "arctanh": "atanh"}  # fmt: skip

EXPRESSIONS = {
    "ADD": "{0} + {1}",
    "SUB": "{0} - {1}",
    "MUL": "{0} * {1}",
    "DIV": "{0} / {1}",
    "NEG": "-{0}",
    "SQ": "{0} * {0}",
    "TWICE": "2.0 * {0}",
    "INV": "1.0 / {0}",
    "POW": "{xp}.{pow}({0}, {1})",
    "CONSTPOW": "{xp}.{pow}({0}, {1})",
    "FMOD": "{xp}.fmod({0}, {1})",
    "FMIN": "{xp}.minimum({0}, {1})",
    "FMAX": "{xp}.maximum({0}, {1})",
    "ATAN2": "{xp}.{atan2}({0}, {1})",
    "COPYSIGN": "{xp}.copysign({0}, {1})",
    "HYPOT": "{xp}.hypot({0}, {1})",
    "LT": "{xp}.where({0} < {1}, one, zero)",
    "LE": "{xp}.where({0} <= {1}, one, zero)",
    "EQ": "{xp}.where({0} == {1}, one, zero)",
    "NE": "{xp}.where({0} != {1}, one, zero)",
    "AND": "{xp}.where(({0} != 0) & ({1} != 0), one, zero)",
    "OR": "{xp}.where(({0} != 0) | ({1} != 0), one, zero)",
    "NOT": "{xp}.where({0} == 0, one, zero)",
    "IF_ELSE_ZERO": "{xp}.where({0} != 0, {1}, zero)",
}
"""Operations of two arguments (and the few others), as the expression that computes them."""

PRELUDE = {
    "numpy": '''import math as _math

import numpy as np

_erf = np.vectorize(_math.erf, otypes=[float])


def _flat(x, rows, cols):
    """The argument as (..., rows * cols), column by column."""
    x = np.asarray(x, dtype=float)
    if (rows, cols) == (1, 1):
        return x[..., None]
    if cols == 1:
        return x
    return np.swapaxes(x, -1, -2).reshape(x.shape[:-2] + (rows * cols,))


def _stack(items, shape, one):
    return np.stack([np.broadcast_to(np.asarray(i, dtype=float), shape) for i in items], axis=-1)


def _zeros(shape, size, one):
    return np.zeros(shape + (size,))


def _shape(out, rows, cols):
    """(..., rows * cols) as the CasADi shape: a scalar, a vector or a matrix."""
    if (rows, cols) == (1, 1):
        return out[..., 0]
    if cols == 1:
        return out
    return np.swapaxes(out.reshape(out.shape[:-1] + (cols, rows)), -1, -2)
''',
    "torch": '''import numpy as np
import torch


def _flat(x, rows, cols):
    """The argument as (..., rows * cols), column by column."""
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))  # a float is float64
    x = x if x.is_floating_point() else x.to(torch.float64)
    if (rows, cols) == (1, 1):
        return x[..., None]
    if cols == 1:
        return x
    return torch.transpose(x, -1, -2).reshape(x.shape[:-2] + (rows * cols,))


def _stack(items, shape, one):
    return torch.stack([(one * i).expand(shape) for i in items], dim=-1)


def _zeros(shape, size, one):
    return torch.zeros(shape + (size,), dtype=one.dtype, device=one.device)


def _shape(out, rows, cols):
    """(..., rows * cols) as the CasADi shape: a scalar, a vector or a matrix."""
    if (rows, cols) == (1, 1):
        return out[..., 0]
    if cols == 1:
        return out
    return torch.transpose(out.reshape(out.shape[:-1] + (cols, rows)), -1, -2)
''',
}
"""What a generated module starts with, by backend: the helpers that give arguments and results
their CasADi shapes."""


def expanded(function: ca.Function) -> ca.Function:
    """The function as a graph of scalar operations (a CasADi ``SXFunction``)."""
    if function.is_a("SXFunction"):
        return function
    try:
        return function.expand()
    except RuntimeError as error:
        raise ValueError(
            f"{function.name()!r} cannot be written as scalar operations: {error}"
        ) from None


def literal(value: float) -> str:
    """A float as Python source, exactly."""
    if math.isnan(value):
        return "float('nan')"
    if math.isinf(value):
        return "float('inf')" if value > 0 else "-float('inf')"
    return repr(float(value))


def source(function: ca.Function, backend: str = "numpy", name: str | None = None) -> str:
    """Python source of ``function`` over numpy arrays or torch tensors.

    The generated function takes one argument per input, with any leading batch dimensions: a
    scalar is a number, a vector an array of shape (..., n) and a matrix one of shape (..., n, m).
    It returns the outputs in the same shapes, a tuple if there are several. Everything is
    computed as the CasADi graph computes it, operation by operation.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend takes one of {BACKENDS}, got {backend!r}")
    f = expanded(function)
    name = name or f.name() or "f"
    torch_ = backend == "torch"
    xp = "torch" if torch_ else "np"
    n_in, n_out = f.n_in(), f.n_out()
    ops = {getattr(ca, n): n[3:] for n in dir(ca) if n.startswith("OP_")}
    lines = [f"def {name}({', '.join(f'x{i}' for i in range(n_in))}):"]
    for i in range(n_in):
        rows, cols = f.size_in(i)
        lines.append(f"    x{i} = _flat(x{i}, {rows}, {cols})")
    batches = ", ".join(f"x{i}.shape[:-1]" for i in range(n_in))
    lines.append(f"    shape = {xp}.broadcast_shapes({batches})" if n_in else "    shape = ()")
    if torch_:
        like = "x0" if n_in else "torch.zeros(())"
        lines.append(f"    one = torch.ones((), dtype={like}.dtype, device={like}.device)")
        lines.append("    zero = torch.zeros_like(one)")
    else:
        lines.append("    one, zero = 1.0, 0.0")
    for k in range(f.n_instructions()):
        op, inputs, out = ops[f.instruction_id(k)], f.instruction_input(k), f.instruction_output(k)
        if op == "INPUT":
            lines.append(f"    w{out[0]} = x{inputs[0]}[..., {inputs[1]}]")
        elif op == "OUTPUT":  # a slot is reused later: the value is kept now
            lines.append(f"    y{out[0]}_{out[1]} = w{inputs[0]}")
        elif op == "CONST":
            value = literal(f.instruction_constant(k))
            lines.append(f"    w{out[0]} = {'one * ' if torch_ else ''}{value}")
        elif op in UNARY:
            fn = UNARY[op]
            fn = TORCH_NAMES.get(fn, fn) if torch_ else fn
            call = "_erf" if op == "ERF" and not torch_ else f"{xp}.{fn}"
            lines.append(f"    w{out[0]} = {call}(w{inputs[0]})")
        elif op in EXPRESSIONS:
            expr = EXPRESSIONS[op].format(
                *(f"w{i}" for i in inputs),
                xp=xp,
                pow="pow" if torch_ else "power",
                atan2="atan2" if torch_ else "arctan2",
            )
            lines.append(f"    w{out[0]} = {expr}")
        else:
            raise NotImplementedError(f"{f.name()!r} uses {op}, which has no {backend} translation")
    returns = []
    for i in range(n_out):
        rows, cols = f.size_out(i)
        sparsity = f.sparsity_out(i)
        values = wrapped([f"y{i}_{j}" for j in range(sparsity.nnz())])
        stack = f"_stack([{values}], shape, one)"
        if sparsity.nnz() == rows * cols:  # dense: the nonzeros are in order
            lines.append(f"    o{i} = {stack}")
        else:  # structural zeros between the nonzeros
            lines.append(f"    o{i} = _zeros(shape, {rows * cols}, one)")
            if sparsity.nnz():
                index = wrapped([str(k) for k in dense_index(sparsity)])
                lines.append(f"    o{i}[..., [{index}]] = {stack}")
        returns.append(f"_shape(o{i}, {rows}, {cols})")
    result = returns[0] if n_out == 1 else f"({', '.join(returns)})"
    lines.append(f"    return {result}")
    head = f'"""The CasADi function {f.name()!r} as {backend} code: no CasADi needed."""\n\n'
    return head + PRELUDE[backend] + "\n\n" + "\n".join(lines) + "\n"


def wrapped(items: list[str]) -> str:
    """Comma-separated ``items`` in lines of about 90 characters, to go inside brackets."""
    return textwrap.fill(", ".join(items), width=88, subsequent_indent="        ")


def dense_index(sparsity: Any) -> list[int]:
    """Where each nonzero of ``sparsity`` sits in the dense column-major vector."""
    rows = sparsity.size1()
    row, col = sparsity.get_triplet()
    return [int(r + c * rows) for r, c in zip(row, col, strict=True)]


def translate(function: ca.Function, backend: str = "numpy", name: str | None = None) -> Callable:
    """``function`` as a Python callable over numpy arrays or torch tensors, from its ``source``
    (kept as the callable's ``source`` attribute)."""
    text = source(function, backend, name)
    name = name or function.name() or "f"
    namespace: dict[str, Any] = {}
    exec(compile(text, f"<{name} as {backend}>", "exec"), namespace)
    fn: Callable = namespace[name]
    fn.source = text  # type: ignore[attr-defined]
    return fn


def export(functions: Mapping[str, ca.Function | None], backend: str) -> SimpleNamespace:
    """The functions of a model, by name, as callables of ``backend`` (``None`` stays ``None``),
    each with its ``source``. ``backend="casadi"`` gives the functions as they are."""
    if backend == "casadi":
        return SimpleNamespace(**dict(functions))
    return SimpleNamespace(
        **{k: None if f is None else translate(f, backend, k) for k, f in functions.items()}
    )
