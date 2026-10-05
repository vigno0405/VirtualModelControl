"""Configurations: experiments in YAML files, a robot template, its controllers and how to run
them, built through the registry."""

from .coordinates import Scope
from .experiment import Experiment, load
from .files import read, write

__all__ = ["Experiment", "Scope", "load", "read", "write"]
