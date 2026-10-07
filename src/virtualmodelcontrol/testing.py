"""Checks of the library's model contract, to call from your own model's tests."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise
from typing import Any

import numpy as np

from .models.kinematic import evaluate_frame, from_dict
from .models.kinematics import Kinematics

__all__ = [
    "check_model",
]


def check_model(
    model: Any,
    at: Iterable[Any] | None = None,
    s: Iterable[float] | None = None,
    *,
    samples: int = 5,
    scale: float = 0.05,
    seed: int = 0,
    energy: bool = True,
) -> dict[str, float]:
    """Check a model, or a robot's mechanism, against the contract; returns the worst error of
    each check, and raises an ``AssertionError`` that lists the checks that failed.

    At the neutral configuration and at ``samples`` random ones (the neutral one plus a step of
    size ``scale`` in every direction), for every site in ``at`` (default: all the model's
    sites) and every arc parameter in ``s``, it checks that everything is finite, also with each
    Param a tenth of its default and ten times it (kept within its bounds), that rotations are
    orthonormal, that the Jacobians and the Hessian agree with finite differences, and that the
    model survives ``to_dict`` and ``from_dict``. Along the ``s`` it checks that the body has no
    jump. A robot mechanism with masses and no dampers must also keep its energy in a
    simulation. For a robot with equations of motion of its own (``models.Equations``) it checks
    that the residual is affine in the acceleration, that the mass matrix is symmetric and
    positive definite, and, when they give an energy, that a simulation never gains energy.
    Building the derivatives of a site takes a moment: name a few sites in ``at`` for a big
    robot.
    """
    robot = model if hasattr(model, "components") else None
    kinematic = robot.model if robot is not None else model
    kin, space, rng = Kinematics(kinematic), kinematic.space, np.random.default_rng(seed)
    spots = [*(kinematic.sites if at is None else at), *(() if s is None else tuple(s))]
    configs = [space.neutral()] + [
        space.integrate(space.neutral(), scale * rng.standard_normal(space.nv))
        for _ in range(samples)
    ]
    worst: dict[str, float] = {}
    failed: list[str] = []

    def note(name: str, error: float, tolerance: float, where: str) -> None:
        worst[name] = max(worst.get(name, 0.0), float(error))
        if not error <= tolerance:  # also true for NaN
            failed.append(f"{name} at {where}: {error:.2e} (limit {tolerance:.0e})")

    for spot in spots:
        for q in configs:
            _derivatives(kin, space, spot, q, note)
    _extremes(kinematic, spots, configs[:2], note)
    if s is not None:
        _continuity(kinematic, [float(x) for x in s], configs[:2], note)
    _serialization(kinematic, kin, spots, configs, note)
    if robot is not None and energy:
        _energy(robot, configs[-1], rng, note)
    if robot is not None and getattr(kinematic, "equations", None) is not None:
        _equations(robot, configs, rng, note, energy)
    if failed:
        raise AssertionError("the model breaks the contract: " + "; ".join(failed))
    return worst


def _derivatives(kin: Kinematics, space: Any, at: Any, q: np.ndarray, note: Any) -> None:
    """Finite, orthonormal, and the Jacobians and the Hessian against finite differences."""
    where = f"{at!r}"
    p, R, J, Jw, H = (np.array(x) for x in kin.functions(at)(q))
    finite = sum(int(not np.isfinite(x).all()) for x in (p, R, J, Jw, H))
    note("finite", finite, 0, where)
    if finite:
        return
    orthonormal = max(np.abs(R @ R.T - np.eye(3)).max(), abs(np.linalg.det(R) - 1.0))
    note("rotation", orthonormal, 1e-6, where)
    h, nv = 1e-6, space.nv
    steps = [(space.integrate(q, h * e), space.integrate(q, -h * e)) for e in np.eye(nv)]
    fd = np.column_stack([(kin.position(a, at) - kin.position(b, at)) / (2 * h) for a, b in steps])
    note("jacobian", np.abs(J - fd).max() / (1.0 + np.abs(J).max()), 1e-5, where)
    spin = []
    for a, b in steps:
        S = (kin.rotation(a, at) - kin.rotation(b, at)) / (2 * h) @ R.T
        spin.append([S[2, 1], S[0, 2], S[1, 0]])
    note(
        "angular_jacobian",
        np.abs(Jw - np.array(spin).T).max() / (1.0 + np.abs(Jw).max()),
        1e-5,
        where,
    )
    if space.nq == space.nv:
        g = 1e-5
        rows = [
            (kin.jacobian(q + g * e, at) - kin.jacobian(q - g * e, at)) / (2 * g)
            for e in np.eye(nv)
        ]
        fd_hessian = np.stack(rows, axis=-1)  # [k, i, j] = ∂J[k, i] / ∂q_j
        hessian = H.reshape(3, nv, nv)
        note(
            "hessian",
            np.abs(hessian - fd_hessian).max() / (1.0 + np.abs(hessian).max()),
            1e-4,
            where,
        )


def _extremes(kinematic: Any, spots: list[Any], configs: list[np.ndarray], note: Any) -> None:
    """The frames stay finite with each Param a tenth of its default and ten times it, kept
    within its bounds."""
    for name, param in kinematic.params.items():
        kept = param.value.copy()
        bounds = [np.asarray(b, dtype=float) for b in param.bounds]
        lower, upper = (np.broadcast_to(b, param.shape) for b in bounds)
        try:
            for factor in (0.1, 10.0):
                value = np.clip(kept * factor, lower, upper)
                if np.array_equal(value, kept):
                    continue
                param.value = value
                for spot in spots:
                    for q in configs:
                        frame = evaluate_frame(kinematic, q, spot)
                        bad = sum(int(not np.isfinite(x).all()) for x in frame)
                        note("finite", bad, 0, f"{spot!r} with {name} x{factor:g}")
        finally:
            param.value = kept


def _continuity(kinematic: Any, grid: list[float], configs: list[np.ndarray], note: Any) -> None:
    """The body has no jump along ``s``: neighboring points are about as far apart everywhere."""
    for q in configs:
        frames = [evaluate_frame(kinematic, q, x) for x in grid]
        points = np.array([p for _, p in frames])
        gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        note("continuity", np.max(gaps) / max(np.median(gaps), 1e-12), 3.0, "the positions along s")
        angles = [
            np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))
            for (a, _), (b, _) in pairwise(frames)
        ]
        note("continuity_rotation", max(angles), 0.5, "the rotations along s")


def _serialization(
    kinematic: Any, kin: Kinematics, spots: list[Any], configs: list[np.ndarray], note: Any
) -> None:
    """The model written by ``to_dict`` and read by ``from_dict`` has the same frames."""
    if not hasattr(kinematic, "to_dict"):
        return
    try:
        clone = Kinematics(from_dict(kinematic.to_dict()))
    except Exception as error:  # not registered, a missing key, a wrong type: all are failures
        note("serialization", np.inf, 0.0, f"to_dict and from_dict: {error!r}")
        return
    gap = 0.0
    for spot in spots:
        for q in configs:
            gap = max(gap, np.abs(kin.position(q, spot) - clone.position(q, spot)).max())
            gap = max(gap, np.abs(kin.rotation(q, spot) - clone.rotation(q, spot)).max())
    note("serialization", gap, 1e-12, "a model rebuilt from its dict")


def _energy(robot: Any, q: np.ndarray, rng: np.random.Generator, note: Any) -> None:
    """A robot with masses and no dampers keeps its energy: the drift of the simulator's steps
    falls with the step, where a force that does work would not let it fall."""
    from .sim.model_plant import ModelPlant

    kinds = [component.kind for component in robot.components.values()]
    if "inertance" not in kinds or "dissipation" in kinds:
        return
    v = 0.1 * rng.standard_normal(robot.model.space.nv)
    drifts = []
    for step in (1e-4, 2.5e-5):
        plant = ModelPlant(robot, q0=q, v0=v, max_step=step)
        start = plant.energy()
        plant.advance(0.5)
        drifts.append(abs(plant.energy() - start))
    note(
        "energy",
        max(drifts[1] - 0.5 * drifts[0], 0.0) / (abs(start) + 1e-12),
        1e-6,
        "half a second",
    )


def _equations(
    robot: Any, configs: list[np.ndarray], rng: np.random.Generator, note: Any, energy: bool
) -> None:
    """Equations of motion given as a function: the mass matrix is symmetric and positive
    definite, and (with an energy, and no source in the robot) the energy never grows."""
    from .dynamics import compile_dynamics
    from .sim.model_plant import ModelPlant

    try:
        dynamics = compile_dynamics(robot)
    except ValueError as error:  # not affine in a, or an inertance next to the equations
        note("equations", np.inf, 0.0, f"the equations of motion: {error}")
        return
    p = dynamics.live_values()
    for q in configs:
        M = np.array(dynamics.mass(q, p))
        scale = max(np.abs(M).max(), 1e-12)
        note("mass_finite", int(not np.isfinite(M).all()), 0, "the mass matrix")
        if not np.isfinite(M).all():
            continue
        note("mass_symmetric", np.abs(M - M.T).max() / scale, 1e-9, "the mass matrix")
        note(
            "mass_positive",
            max(-np.linalg.eigvalsh((M + M.T) / 2).min(), 0.0) / scale,
            0.0,
            "the mass matrix",
        )
    kinds = [component.kind for component in robot.components.values()]
    if dynamics.energy is None or "source" in kinds or not energy:
        return
    v = 0.1 * rng.standard_normal(robot.model.space.nv)
    plant = ModelPlant(robot, q0=configs[-1], v0=v, max_step=1e-4)
    start = plant.energy()
    plant.advance(0.5)
    note(
        "energy_growth",
        max(plant.energy() - start, 0.0) / (abs(start) + 1e-12),
        1e-6,
        "half a second",
    )
