"""An experiment from a configuration: the robot, its controllers, and how to run them."""

from __future__ import annotations

import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import robots  # noqa: F401 (registers the robot templates)
from ..compiler import compile
from ..control import Schedule, ScheduledController, VMCController
from ..core.params import Param
from ..mechanisms import Coordinate, Mechanism
from ..sim import RECORDS, ModelPlant, RunLog, SimClock, run
from ..system import VirtualMechanismSystem
from . import files
from .controllers import build_controller, build_element
from .coordinates import Scope
from .spec import Location, check_keys, template, where

SECTIONS = ("robot", "coordinates", "controller", "swaps", "experiment")
SETTINGS = ("plant", "rate", "duration", "runtime", "output", "schedule", "z0", "run")
RUN = ("name", "folder", "overwrite", "record")
PLANTS = {"simulation": ("type", "dynamics", "elements", "q0", "v0", "max_step")}


class Experiment:
    """A robot, its controllers and how to run them, from a configuration.

    The sections: ``robot`` (a template and its arguments), ``coordinates`` (named, shared by
    the controllers), ``controller``, ``swaps`` (named controllers to swap to) and
    ``experiment`` (the simulated plant, the rate [Hz] and duration [s], the output stage,
    schedules, and ``run``: where its log goes and what it records). The plant's ``dynamics``
    template and ``elements`` (contacts, say) go on the simulated robot only. ``run`` runs it;
    ``save`` writes it back with the current values of its Params. ``folder`` is where relative
    log folders start (the configuration file's folder, for ``load``; else the working
    directory).
    """

    def __init__(
        self, spec: dict[str, Any], *, folder: str | os.PathLike[str] | None = None
    ) -> None:
        self.spec = copy.deepcopy(dict(spec))
        self.folder = None if folder is None else Path(folder)
        check_keys(self.spec, (), SECTIONS, required=("robot", "controller"))
        self.settings = check_keys(self.spec.get("experiment") or {}, ("experiment",), SETTINGS)
        self.plant_settings = _plant_settings(self.settings.get("plant", {"type": "simulation"}))
        self.run_settings = _run_settings(self.settings.get("run"))
        self._tracked: list[tuple[Location, Param]] = []
        self.robot: Mechanism = template("robot", self.spec["robot"], ("robot",))()
        dynamics = self.plant_settings.get("dynamics")
        if dynamics is not None:
            template("dynamics", dynamics, ("experiment", "plant", "dynamics"))(self.robot)

        names: dict[str, Coordinate] = {}
        scope = Scope(self.robot, names, self._tracked)
        for key, coordinate in (self.spec.get("coordinates") or {}).items():
            names[key] = scope.build(coordinate, ("coordinates", key))
        for key, element in (self.plant_settings.get("elements") or {}).items():
            path = ("experiment", "plant", "elements", key)
            self.robot.add(key, build_element(element, scope, path))
        main = self.spec["controller"]
        self.name: str = main.get("name", "ctrl") if isinstance(main, dict) else "ctrl"
        self.mechanisms = {
            self.name: build_controller(
                main, self.name, self.robot, names, self._tracked, ("controller",), named=True
            )
        }
        for key, swap in (self.spec.get("swaps") or {}).items():
            if key in self.mechanisms:
                raise ValueError(f"swaps.{key}: the controller is named {key!r} too")
            self.mechanisms[key] = build_controller(
                swap, key, self.robot, names, self._tracked, ("swaps", key)
            )

        output, runtime = self._output(), self.settings.get("runtime", ())
        self.controllers = {
            key: VMCController(compile(VirtualMechanismSystem(self.robot, m), runtime), output)
            for key, m in self.mechanisms.items()
        }
        self.controller: Any = self._scheduled()
        self._plant: ModelPlant | None = None

    @property
    def mechanism(self) -> Mechanism:
        """The controller's mechanism (the one the experiment starts with)."""
        return self.mechanisms[self.name]

    @property
    def plant(self) -> ModelPlant:
        """The simulated robot, built when first used."""
        settings = self.plant_settings
        if self._plant is None:
            self._plant = ModelPlant(
                self.robot,
                settings.get("q0"),
                settings.get("v0"),
                max_step=settings.get("max_step", 1e-3),
            )
        return self._plant

    def run(self) -> RunLog:
        """Run the experiment from its start and return its log (the same run each time). With
        ``run`` settings, the log is also saved as ``<folder>/<name>.npz`` with this configuration
        in it; a name already taken stops the run before it starts, unless ``overwrite``."""
        duration = self.settings.get("duration")
        if "rate" not in self.settings or duration is None:
            raise ValueError("experiment: an experiment needs its `rate` and `duration` to run")
        path = self.log_path()
        overwrite = bool(self.run_settings.get("overwrite", False))
        if path is not None and path.exists() and not overwrite:
            raise FileExistsError(
                f"{path} exists: give the run another name, or `overwrite: true` in experiment.run"
            )
        configuration = files.dumps(self.to_dict())  # the values the run starts with
        self.plant.reset()
        clock = SimClock(1.0 / self.settings["rate"])
        record = self.run_settings.get("record", ())
        log = run(self.plant, self.controller, clock, duration, z0=self.z0(), record=record)
        log.meta["configuration"] = configuration
        if path is not None:
            log.save(path, overwrite=overwrite)
        return log

    def log_path(self) -> Path | None:
        """Where ``run`` saves the log: ``<folder>/<name>.npz`` (``name`` by default the start
        time), or None without ``run`` settings."""
        if not self.run_settings:
            return None
        name = str(self.run_settings.get("name") or datetime.now().strftime("run-%Y%m%d-%H%M%S"))
        folder = Path(self.run_settings.get("folder", "."))
        if not folder.is_absolute():
            folder = (self.folder or Path.cwd()) / folder
        return folder / f"{name.removesuffix('.npz')}.npz"

    def to_dict(self) -> dict[str, Any]:
        """The configuration, with the current values of the controllers' Params."""
        out = copy.deepcopy(self.spec)
        for path, param in self._tracked:
            node = out
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = param.value.tolist()
        return out

    def save(self, path: str | os.PathLike[str]) -> Any:
        """Write the configuration as YAML, with the Params' current values; returns the path."""
        return files.write(self.to_dict(), path)

    def _output(self) -> list[Any]:
        stages: list[Any] = []
        for i, spec in enumerate(self.settings.get("output") or []):
            path = ("experiment", "output", i)
            check_keys(spec, path, None, required=("type",))
            args = {k: v for k, v in spec.items() if k != "type"}
            made = template("output", {"template": spec["type"], **args}, path)()
            stages.extend(made if isinstance(made, list) else [made])
        return stages

    def _scheduled(self) -> Any:
        schedules, swaps = [], []
        for i, entry in enumerate(self.settings.get("schedule") or []):
            path = ("experiment", "schedule", i)
            if isinstance(entry, dict) and "swap" in entry:
                check_keys(entry, path, ("swap", "at", "duration"), required=("at", "duration"))
                if entry["swap"] not in self.controllers:
                    raise KeyError(
                        f"{where((*path, 'swap'))}: no controller named {entry['swap']!r}; "
                        f"known: {list(self.controllers)}"
                    )
                target = self.controllers[entry["swap"]]
                swaps.append((float(entry["at"]), target, float(entry["duration"])))
                continue
            check_keys(
                entry, path, ("param", "points", "interpolation"), required=("param", "points")
            )
            try:
                schedule = Schedule(
                    entry["param"], entry["points"], entry.get("interpolation", "linear")
                )
            except ValueError as exc:
                raise ValueError(f"{where(path)}: {exc}") from None
            schedules.append(schedule)
        main = self.controllers[self.name]
        if not schedules and not swaps:
            return main
        try:
            return ScheduledController(main, schedules, swaps)
        except KeyError as exc:
            raise KeyError(f"experiment.schedule: {exc.args[0]}") from None

    def z0(self) -> Any:
        """The initial virtual state: a list, a function of the first reading, or None."""
        z0 = self.settings.get("z0")
        if z0 is None or isinstance(z0, list):
            return z0
        return template("initial_state", z0, ("experiment", "z0"))


def _plant_settings(spec: Any) -> dict[str, Any]:
    path = ("experiment", "plant")
    check_keys(spec, path, None, required=("type",))
    if spec["type"] not in PLANTS:
        raise ValueError(
            f"{where((*path, 'type'))}: unknown plant {spec['type']!r}; known: {list(PLANTS)}"
        )
    return check_keys(spec, path, PLANTS[spec["type"]])


def _run_settings(spec: Any) -> dict[str, Any]:
    if spec is None:
        return {}
    path = ("experiment", "run")
    check_keys(spec, path, RUN)
    unknown = set(spec.get("record", ())) - set(RECORDS)
    if unknown:
        raise ValueError(f"{where((*path, 'record'))}: {sorted(unknown)}; known: {list(RECORDS)}")
    return spec


def load(source: str | os.PathLike[str] | dict[str, Any]) -> Experiment:
    """An experiment from a YAML file (its folder is where relative log folders start), or from
    its configuration as a dict."""
    if isinstance(source, dict):
        return Experiment(source)
    return Experiment(files.read(source), folder=Path(source).resolve().parent)
