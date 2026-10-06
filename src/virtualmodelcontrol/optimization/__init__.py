"""Optimization of virtual mechanisms: plan the closed loop's motion and tune the Params."""

from .asktell import CMAES, Bayes, ExtremumSeeking, Grid, Random, bounds_of, tune
from .collocation import Collocation
from .equilibrium import Equilibrium
from .horizon import MovingHorizon
from .mpc import MPC
from .nlp import NLP
from .problem import Problem
from .result import Result
from .search import Search, search_references, sphere_points
from .shooting import Shooting, ShootingTrajectory
from .solver import CONVERGED, IPOPT, PRESETS
from .tank import TankBudget
from .terms import Bound, Cost, Effort, Period, Sparsity, Term
from .trajectory import Trajectory

__all__ = [
    "CMAES",
    "CONVERGED",
    "IPOPT",
    "MPC",
    "NLP",
    "PRESETS",
    "Bayes",
    "Bound",
    "Collocation",
    "Cost",
    "Effort",
    "Equilibrium",
    "ExtremumSeeking",
    "Grid",
    "MovingHorizon",
    "Period",
    "Problem",
    "Random",
    "Result",
    "Search",
    "Shooting",
    "ShootingTrajectory",
    "Sparsity",
    "TankBudget",
    "Term",
    "Trajectory",
    "bounds_of",
    "search_references",
    "sphere_points",
    "tune",
]
