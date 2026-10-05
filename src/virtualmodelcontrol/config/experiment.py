"""An experiment from a configuration: the robot, its controllers, and how to run them."""

from __future__ import annotations

import copy
import os
from typing import Any

from .. import robots  # noqa: F401 (registers the robot templates)
from ..compiler import compile
from ..control import Schedule, ScheduledController, VMCController
from ..core.params import Param
from ..mechanisms import Coordinate, Mechanism
from ..sim import ModelPlant, RunLog, SimClock, WallClock, run
from ..system import VirtualMechanismSystem
from . import files
from .controllers import build_controller, build_element
from .coordinates import Scope
from .spec import Path, check_keys, template, where

SECTIONS = ("robot", "coordinates", "controller", "swaps", "experiment")
SETTINGS = ("plant", "rate", "duration", "runtime", "output", "schedule", "z0")
PLANTS = {
    "simulation": ("type", "dynamics", "elements", "q0", "v0", "max_step"),
    "dynamixel": ("type", "hardware", "port", "watchdog", "home"),
    "ros": ("type", "hardware", "stale", "name"),
}


class Experiment:
    """A robot, its controllers and how to run them, from a configuration.

    The sections: ``robot`` (a template and its arguments), ``coordinates`` (named, shared by
    the controllers), ``controller``, ``swaps`` (named controllers to swap to) and
    ``experiment`` (the plant, the rate [Hz] and duration [s], the output stage, schedules). A
    simulation's ``dynamics`` template and ``elements`` (contacts, say) go on the simulated robot
    only. ``run`` runs it; ``save`` writes it back with the current values of its Params.
    """

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = copy.deepcopy(dict(spec))
        check_keys(self.spec, (), SECTIONS, required=("robot", "controller"))
        self.settings = check_keys(self.spec.get("experiment") or {}, ("experiment",), SETTINGS)
        self.plant_settings = _plant_settings(self.settings.get("plant", {"type": "simulation"}))
        self._record: list[tuple[Path, Param]] = []
        self.robot: Mechanism = template("robot", self.spec["robot"], ("robot",))()
        dynamics = self.plant_settings.get("dynamics")
        if self.plant_settings["type"] == "simulation" and dynamics is not None:
            template("dynamics", dynamics, ("experiment", "plant", "dynamics"))(self.robot)

        names: dict[str, Coordinate] = {}
        scope = Scope(self.robot, names, self._record)
        for key, coordinate in (self.spec.get("coordinates") or {}).items():
            names[key] = scope.build(coordinate, ("coordinates", key))
        for key, element in (self.plant_settings.get("elements") or {}).items():
            path = ("experiment", "plant", "elements", key)
            self.robot.add(key, build_element(element, scope, path))
        main = self.spec["controller"]
        self.name: str = main.get("name", "ctrl") if isinstance(main, dict) else "ctrl"
        self.mechanisms = {
            self.name: build_controller(
                main, self.name, self.robot, names, self._record, ("controller",), named=True
            )
        }
        for key, swap in (self.spec.get("swaps") or {}).items():
            if key in self.mechanisms:
                raise ValueError(f"swaps.{key}: the controller is named {key!r} too")
            self.mechanisms[key] = build_controller(
                swap, key, self.robot, names, self._record, ("swaps", key)
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
        """The simulated robot of a simulation, built when first used."""
        settings = self.plant_settings
        if settings["type"] != "simulation":
            raise ValueError("only a simulation has its plant here; `run` opens the robot's")
        if self._plant is None:
            self._plant = ModelPlant(
                self.robot,
                settings.get("q0"),
                settings.get("v0"),
                max_step=settings.get("max_step", 1e-3),
            )
        return self._plant

    def run(self, *, bus: Any = None) -> RunLog:
        """Run the experiment and return its log: a simulation from its start (the same run each
        time), or the real robot in real time. ``bus`` replaces a Dynamixel plant's serial bus,
        for a dry run with ``hardware.FakeBus``."""
        kind, duration = self.plant_settings["type"], self.settings.get("duration")
        z0 = self._z0()
        if kind == "simulation":
            if "rate" not in self.settings or duration is None:
                raise ValueError("experiment: a simulation needs its `rate` and `duration`")
            self.plant.reset()
            clock = SimClock(1.0 / self.settings["rate"])
            return run(self.plant, self.controller, clock, duration, z0=z0)
        path = ("experiment", "plant", "hardware")
        profile = template("hardware", self.plant_settings["hardware"], path)()
        if "port" in self.plant_settings:
            profile = profile.replace(port=self.plant_settings["port"])
        dt = 1.0 / self.settings.get("rate", profile.rate)
        if kind == "dynamixel":
            from ..hardware import DynamixelPlant

            motors = DynamixelPlant(profile, bus, watchdog=self.plant_settings.get("watchdog", 0.1))
            if self.plant_settings.get("home", False) and not motors.home():
                motors.close()
                raise RuntimeError("the motors did not reach their home pose")
            with motors:
                return run(motors, self.controller, WallClock(dt), duration, z0=z0)
        from ..ros.plant import RosPlant

        ros = RosPlant(profile, name=self.plant_settings.get("name", "vmc_plant"))
        try:
            ros.wait()
            wall = WallClock(dt, stale=self.plant_settings.get("stale", 0.05))
            return run(ros, self.controller, wall, duration, z0=z0)
        finally:
            ros.close()

    def to_dict(self) -> dict[str, Any]:
        """The configuration, with the current values of the controllers' Params."""
        out = copy.deepcopy(self.spec)
        for path, param in self._record:
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

    def _z0(self) -> Any:
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
    required = () if spec["type"] == "simulation" else ("hardware",)
    return check_keys(spec, path, PLANTS[spec["type"]], required)


def load(source: str | os.PathLike[str] | dict[str, Any]) -> Experiment:
    """An experiment from a YAML file, or from its configuration as a dict."""
    return Experiment(source if isinstance(source, dict) else files.read(source))
