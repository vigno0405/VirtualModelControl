"""Gradient-free tuning of Params by ask and tell: you run the episodes, it picks the next."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike

from ..core.params import ParamSet
from .builder import param_bounds


class _Searcher:
    """What every searcher keeps: ``history`` of (candidate, cost), and the ``best`` so far."""

    def __init__(self) -> None:
        self.history: list[tuple[np.ndarray, float]] = []
        self.best: np.ndarray | None = None
        self.best_cost = np.inf

    def ask(self) -> list[np.ndarray]:
        """The next candidates to try (an empty list when there are none left)."""
        raise NotImplementedError

    def tell(self, candidates: Sequence[ArrayLike], costs: Sequence[float]) -> None:
        """Give the cost of each candidate that was asked, in the same order."""
        if len(candidates) != len(costs):
            raise ValueError(f"{len(candidates)} candidates but {len(costs)} costs")
        for x, cost in zip(candidates, costs, strict=True):
            self.history.append((np.array(x, dtype=float), float(cost)))
            if cost < self.best_cost:
                self.best, self.best_cost = self.history[-1][0], float(cost)


class Grid(_Searcher):
    """Every point of a regular grid between the bounds, asked once; ``points`` per dimension."""

    def __init__(self, lower: ArrayLike, upper: ArrayLike, points: int | Sequence[int]) -> None:
        super().__init__()
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        counts = np.broadcast_to(points, lower.shape)
        axes = [np.linspace(a, b, n) for a, b, n in zip(lower, upper, counts, strict=True)]
        self._points = list(np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, lower.size))

    def ask(self) -> list[np.ndarray]:
        """All the grid points the first time, then nothing."""
        points, self._points = self._points, []
        return points


class Random(_Searcher):
    """``size`` uniform samples inside the bounds at every ask."""

    def __init__(self, lower: ArrayLike, upper: ArrayLike, size: int = 8, seed: int = 0) -> None:
        super().__init__()
        self.lower, self.upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        self.size, self._rng = size, np.random.default_rng(seed)

    def ask(self) -> list[np.ndarray]:
        """New samples."""
        return list(self._rng.uniform(self.lower, self.upper, (self.size, self.lower.size)))


class CMAES(_Searcher):
    """(mu/mu_w, lambda)-CMA-ES from the mean ``x0`` with step size ``sigma``; default lambda is
    4 + floor(3 ln n). Candidates are clipped into the bounds."""

    def __init__(
        self,
        x0: ArrayLike,
        sigma: float,
        lower: ArrayLike | None = None,
        upper: ArrayLike | None = None,
        size: int | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.lower = -np.inf if lower is None else np.asarray(lower, dtype=float)
        self.upper = np.inf if upper is None else np.asarray(upper, dtype=float)
        self.mean = np.clip(np.asarray(x0, dtype=float), self.lower, self.upper)
        n = self.mean.size
        self.sigma, self._rng, self._generation = float(sigma), np.random.default_rng(seed), 0
        self.size = 4 + int(3 * np.log(n)) if size is None else size
        mu = self.size // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self._w = w / w.sum()
        self._mueff = m = 1.0 / np.sum(self._w**2)
        self._cc = (4 + m / n) / (n + 4 + 2 * m / n)
        self._cs = (m + 2) / (n + m + 5)
        self._c1 = 2 / ((n + 1.3) ** 2 + m)
        self._cmu = min(1 - self._c1, 2 * (m - 2 + 1 / m) / ((n + 2) ** 2 + m))
        self._damps = 1 + 2 * max(0.0, np.sqrt((m - 1) / (n + 1)) - 1) + self._cs
        self._chi = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        self.cov, self._pc, self._ps = np.eye(n), np.zeros(n), np.zeros(n)

    def _axes(self) -> tuple[np.ndarray, np.ndarray]:
        """Principal axes B (columns) and standard deviations D of the covariance."""
        eigenvalues, B = np.linalg.eigh(self.cov)
        return B, np.sqrt(np.maximum(eigenvalues, 1e-20))

    def ask(self) -> list[np.ndarray]:
        """The population: the mean plus sigma times samples of the covariance, clipped."""
        B, D = self._axes()
        steps = (self._rng.standard_normal((self.size, self.mean.size)) * D) @ B.T
        return list(np.clip(self.mean + self.sigma * steps, self.lower, self.upper))

    def tell(self, candidates: Sequence[ArrayLike], costs: Sequence[float]) -> None:
        """Record the costs, then move the mean, the step size and the covariance."""
        super().tell(candidates, costs)
        n, w = self.mean.size, self._w
        B, D = self._axes()
        best = np.argsort(costs)[: w.size]
        y = (np.asarray(candidates, dtype=float)[best] - self.mean) / self.sigma
        ybar = w @ y
        self.mean = self.mean + self.sigma * ybar
        self._ps = (1 - self._cs) * self._ps + np.sqrt(self._cs * (2 - self._cs) * self._mueff) * (
            B @ ((B.T @ ybar) / D)
        )
        self._generation += 1
        decay = np.sqrt(1 - (1 - self._cs) ** (2 * self._generation))
        stalled = np.linalg.norm(self._ps) / decay / self._chi >= 1.4 + 2 / (n + 1)
        self._pc = (1 - self._cc) * self._pc + (not stalled) * np.sqrt(
            self._cc * (2 - self._cc) * self._mueff
        ) * ybar
        self.cov = (
            (1 - self._c1 - self._cmu) * self.cov
            + self._c1
            * (np.outer(self._pc, self._pc) + stalled * self._cc * (2 - self._cc) * self.cov)
            + self._cmu * (y.T * w) @ y
        )
        self.sigma *= np.exp(self._cs / self._damps * (np.linalg.norm(self._ps) / self._chi - 1))


S = TypeVar("S", bound=_Searcher)


def tune(searcher: S, evaluate: Callable[[np.ndarray], float], rounds: int) -> S:
    """``rounds`` times: ask, evaluate each candidate with ``evaluate(x)``, tell."""
    for _ in range(rounds):
        candidates = searcher.ask()
        searcher.tell(candidates, [evaluate(x) for x in candidates])
    return searcher


def bounds_of(params: ParamSet, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower and upper bounds and current values of the named Params as flat vectors."""
    bounds = [param_bounds(params[name]) for name in names]
    lower, upper = (np.concatenate([b[i] for b in bounds]) for i in range(2))
    return lower, upper, params.vector(names)
