"""Configurations: an experiment in a file runs as the same experiment written in Python, every
coordinate kind builds what Python builds, and saved files carry the current values."""

import copy
import re
from pathlib import Path

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.config import files
from virtualmodelcontrol.control import Schedule, ScheduledController
from virtualmodelcontrol.robots import adapt, helyx, turtle

CONFIGS = Path(__file__).parents[1] / "data" / "configs"
FILES = sorted(CONFIGS.glob("*.yaml"))
rng = np.random.default_rng(11)


def assert_same_run(a, b):
    assert sorted(a) == sorted(b)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_the_soft_arm_file_runs_as_the_same_python():
    from_file = vmc.config.load(CONFIGS / "soft-arm.yaml").run().arrays()

    arm = helyx.add_dynamics(helyx.arm("145-145-145"))
    tip, middle = arm.point(s=1.0), arm.point(s=0.5)
    ctrl = vmc.Mechanism("ctrl")
    goal = vmc.Ref("goal", value=[0.0, 0.0, 0.435])
    ctrl.add("reach", vmc.LinearSpring(tip - goal, 300.0))
    ctrl.add("push", vmc.GaussianSpring(middle - [0.05, 0.0, 0.2], 0.0, 0.05))
    ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    soft = vmc.Mechanism("soft")
    soft.add("reach", vmc.TanhSpring(tip - [0.1, 0.0, 0.40], 100.0, 1.0))
    soft.add("damp", vmc.LinearDamper(tip, 5.0))
    soft.add("gravity", vmc.GravityCompensation(arm))
    first, second = (
        vmc.VMCController(
            vmc.compile(vmc.VirtualMechanismSystem(arm, m)), output=helyx.output_stage()
        )
        for m in (ctrl, soft)
    )
    schedules = [
        Schedule("ctrl.push.strength", [(0.0, 0.0), (0.3, 800.0)]),
        Schedule("ctrl.reach.goal", [(0.0, [0.0, 0.0, 0.435]), (0.4, [0.1, 0.0, 0.40])]),
    ]
    controller = ScheduledController(first, schedules, swaps=[(0.6, second, 0.2)])
    plant = vmc.sim.ModelPlant(arm)
    in_python = vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 330), T=1.0).arrays()
    assert_same_run(from_file, in_python)


def test_the_finger_file_runs_as_the_same_python():
    from_file = vmc.config.load(CONFIGS / "finger.yaml").run().arrays()

    finger = adapt.add_dynamics(adapt.finger(), damping=0.01)
    tip = finger.point("tip")
    table = vmc.PlaneDistance(tip, [0.0, 0.0, -1.0], [0.0, 0.0, 0.06])
    finger.add("table", vmc.ContactSpring(table, 1e4))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("press", vmc.LinearSpring(tip - [0.0, 0.05, 0.065], 100.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 1.0))
    ctrl.add("limits", adapt.joint_limit_spring(finger, 1.0))
    ctrl.add("gravity", vmc.GravityCompensation(finger))
    system = vmc.VirtualMechanismSystem(finger, ctrl)
    controller = vmc.VMCController(vmc.compile(system), output=adapt.output_stage())
    plant = vmc.sim.ModelPlant(finger, q0=[0.8, 0.8], max_step=1e-4)
    in_python = vmc.sim.run(plant, controller, vmc.sim.SimClock(1 / 500), T=0.4).arrays()
    assert_same_run(from_file, in_python)


def test_the_turtle_file_runs_as_the_same_python():
    experiment = vmc.config.load(CONFIGS / "turtle.yaml")
    from_file = experiment.run().arrays()

    robot = turtle.robot()
    robot.add("inertia", vmc.Inertance(robot.joint([0, 1]), 1e-3))
    robot.add("friction", vmc.LinearDamper(robot.joint([0, 1]), 1e-2))
    ctrl = turtle.controller(robot, speed=2.0, ramp_time=0.5, stiffness=0.5, damping=0.01)
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    plant = vmc.sim.ModelPlant(robot, q0=[0.3, 0.0])
    clock = vmc.sim.SimClock(0.01)
    in_python = vmc.sim.run(plant, controller, clock, T=3.0, z0=turtle.initial_state).arrays()
    assert_same_run(from_file, in_python)
    np.testing.assert_array_equal(from_file["z"][0], [0.3, 0.0])  # started at the left crank
    assert_same_run(experiment.run().arrays(), from_file)  # a second run is the same run


@pytest.mark.parametrize("path", FILES, ids=lambda path: path.stem)
def test_every_file_survives_a_round_trip(path, tmp_path):
    experiment = vmc.config.load(path)
    assert experiment.to_dict() == files.read(path)  # nothing changed: the file as written
    again = vmc.config.load(experiment.save(tmp_path / path.name))
    assert again.to_dict() == experiment.to_dict()
    assert list(again.mechanism.params) == list(experiment.mechanism.params)
    robot = experiment.robot
    angles, rates = (
        (robot.space.nq, robot.space.nv)
        if robot.actuation is None
        else robot.actuation.motor_sizes(robot.space)
    )
    meas = vmc.Signals(
        0.0,
        motor_position=rng.uniform(-0.3, 0.3, angles),
        motor_velocity=rng.uniform(-0.3, 0.3, rates),
    )
    np.testing.assert_array_equal(
        again.controller.step(0.0, meas)["motor_torque"],
        experiment.controller.step(0.0, meas)["motor_torque"],
    )


COORDINATES = """
robot: {template: helyx.arm, geometry: "145-145-145"}
coordinates:
  tip: {point: {s: 1.0, offset: [0.0, 0.01, 0.0]}}
  site: {point: tip}
  base: {point: {at: seg1}}
controller:
  states:
    follower: {dim: 3, unit: m, initial: [0.0, 0.0, 0.7]}
  elements:
    a: {type: linear_spring, coordinate: {difference: [[0.1, 0.0, 0.4], tip]}, stiffness: 1.0}
    b:
      type: linear_spring
      coordinate:
        projection: {of: {difference: [site, [0.0, 0.0, 0.4]]}, direction: [1.0, 1.0, 0.0]}
      stiffness: 2.0
    c: {type: linear_spring, coordinate: {norm: {difference: [tip, base]}}, stiffness: 3.0}
    d: {type: linear_spring, coordinate: {slice: {of: tip, index: [0, 2]}}, stiffness: 4.0}
    e:
      type: linear_spring
      coordinate:
        stack: [{joint: 0}, {joint: [3, 4]}, {joint: {start: 6, stop: 9, step: 2}}]
      stiffness: 5.0
    f:
      type: contact_spring
      coordinate:
        plane_distance: {point: tip, normal: [0.0, 0.0, -1.0], origin: [0.0, 0.0, 0.3]}
      stiffness: 100.0
    g:
      type: contact_spring
      coordinate:
        sphere_distance: {point: site, center: [0.0, 0.0, 0.4], radius: 0.05}
      stiffness: 100.0
    h: {type: linear_spring, coordinate: {difference: [tip, {state: follower}]}, stiffness: 6.0}
    i:
      type: linear_spring
      coordinate:
        difference: [{slice: {of: tip, index: 1}}, {ref: {name: level, value: [0.0]}}]
      stiffness: 7.0
    j: {type: constrained_linear_spring, coordinate: tip, stiffness: 8.0, normal: [0, 1, 0]}
    k:
      type: contact_spring
      coordinate:
        box_distance: {point: tip, center: [0.0, 0.0, 0.5], half_sizes: [0.02, 0.03, 0.04]}
      stiffness: 100.0
    l:
      type: contact_spring
      coordinate:
        capsule_distance: {point: tip, a: [0.0, 0.0, 0.5], b: [0.1, 0.0, 0.5], radius: 0.01}
      stiffness: 100.0
    m:
      type: contact_spring
      coordinate:
        cylinder_distance:
          point: tip
          center: [0.0, 0.0, 0.5]
          axis: [0.0, 0.0, 1.0]
          radius: 0.02
          half_height: 0.1
      stiffness: 100.0
    n: {type: linear_spring, coordinate: {sum: [site, [0.0, 0.0, 0.05]]}, stiffness: 9.0}
    o: {type: linear_spring, coordinate: {rotation: {at: tip}}, stiffness: 10.0}
    p:
      type: linear_spring
      coordinate:
        orientation_error: {at: tip, goal: [0.1, 0.0, 0.2]}
      stiffness: 11.0
    q:
      type: linear_spring
      coordinate:
        in_frame: {of: site, at: tip}
      stiffness: [1.0, 2.0, 3.0]
    r: {type: linear_spring, coordinate: {from_frame: {of: site, at: seg1}}, stiffness: 12.0}
    s: {type: diode_damper, coordinate: site, damping: 0.5, sign: -1.0}
    t:
      type: force_source
      coordinate: site
      force: [0.1, 0.0, 0.0]
      max_force: 2.0
      max_power: 5.0
    mass: {type: inertance, coordinate: follower, inertance: 0.05}
"""


def test_every_coordinate_kind_builds_what_python_builds(tmp_path):
    path = tmp_path / "coordinates.yaml"
    path.write_text(COORDINATES)
    experiment = vmc.config.load(path)

    arm = helyx.arm("145-145-145")
    tip, site, base = arm.point(s=1.0, offset=[0.0, 0.01, 0.0]), arm.point("tip"), arm.point("seg1")
    ctrl = vmc.Mechanism("ctrl")
    follower = ctrl.add_state("follower", dim=3, unit="m", initial=[0.0, 0.0, 0.7])
    ctrl.add("a", vmc.LinearSpring([0.1, 0.0, 0.4] - tip, 1.0))
    ctrl.add("b", vmc.LinearSpring(vmc.Projection(site - [0.0, 0.0, 0.4], [1.0, 1.0, 0.0]), 2.0))
    ctrl.add("c", vmc.LinearSpring(vmc.Norm(tip - base), 3.0))
    ctrl.add("d", vmc.LinearSpring(tip[[0, 2]], 4.0))
    joints = vmc.Stack(arm.joint(0), arm.joint([3, 4]), arm.joint(slice(6, 9, 2)))
    ctrl.add("e", vmc.LinearSpring(joints, 5.0))
    ctrl.add("f", vmc.ContactSpring(vmc.PlaneDistance(tip, [0, 0, -1.0], [0, 0, 0.3]), 100.0))
    ctrl.add("g", vmc.ContactSpring(vmc.SphereDistance(site, [0.0, 0.0, 0.4], 0.05), 100.0))
    ctrl.add("h", vmc.LinearSpring(tip - follower, 6.0))
    ctrl.add("i", vmc.LinearSpring(tip[1] - vmc.Ref("level", value=[0.0]), 7.0))
    ctrl.add("j", vmc.ConstrainedLinearSpring(tip, 8.0, normal=[0, 1, 0]))
    ctrl.add("k", vmc.ContactSpring(vmc.BoxDistance(tip, [0, 0, 0.5], [0.02, 0.03, 0.04]), 100.0))
    capsule = vmc.CapsuleDistance(tip, [0, 0, 0.5], [0.1, 0, 0.5], 0.01)
    ctrl.add("l", vmc.ContactSpring(capsule, 100.0))
    cylinder = vmc.CylinderDistance(tip, [0, 0, 0.5], [0, 0, 1.0], 0.02, 0.1)
    ctrl.add("m", vmc.ContactSpring(cylinder, 100.0))
    ctrl.add("n", vmc.LinearSpring(site + np.array([0.0, 0.0, 0.05]), 9.0))
    ctrl.add("o", vmc.LinearSpring(vmc.FrameRotation(arm.model, "tip"), 10.0))
    error = vmc.OrientationError(arm.model, "tip", goal=[0.1, 0.0, 0.2])
    ctrl.add("p", vmc.LinearSpring(error, 11.0))
    ctrl.add("q", vmc.LinearSpring(vmc.InFrame(site, arm.model, "tip"), [1.0, 2.0, 3.0]))
    ctrl.add("r", vmc.LinearSpring(vmc.FromFrame(site, arm.model, "seg1"), 12.0))
    ctrl.add("s", vmc.DiodeDamper(site, 0.5, sign=-1.0))
    ctrl.add("t", vmc.ForceSource(site, [0.1, 0.0, 0.0], max_force=2.0, max_power=5.0))
    ctrl.add("mass", vmc.Inertance(follower, 0.05))
    law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))

    assert list(experiment.mechanism.params) == list(ctrl.params)
    from_file = experiment.controllers["ctrl"].compiled
    np.testing.assert_array_equal(from_file.z0, law.z0)
    p = law.live_values()
    for _ in range(5):
        q, v = rng.uniform(-0.03, 0.03, 9), rng.uniform(-0.1, 0.1, 9)
        positions = np.array([0.0, 0.0, 0.7]) + rng.uniform(-0.1, 0.1, 3)
        z = np.concatenate([positions, rng.uniform(-1.0, 1.0, 3)])
        for a, b in zip(from_file.law(q, v, z, p, 0.0), law.law(q, v, z, p, 0.0), strict=True):
            np.testing.assert_array_equal(np.array(a), np.array(b))


def test_saved_files_carry_the_current_values(tmp_path):
    path = tmp_path / "coordinates.yaml"
    path.write_text(COORDINATES)
    experiment = vmc.config.load(path)
    elements = experiment.mechanism.components
    elements["a"].stiffness.value = 1.5
    elements["a"].coord.a.param.value = [0.2, 0.0, 0.3]  # the plain list on the left
    elements["a"].coord.b.s.value = 0.9  # the named tip's arc parameter
    elements["a"].coord.b.offset.value = [0.0, 0.02, 0.0]
    elements["b"].coord.direction.value = [0.0, 1.0, 1.0]
    elements["f"].coord.origin.value = [0.0, 0.0, 0.25]
    elements["g"].coord.radius.value = 0.06
    elements["k"].coord.half_sizes.value = [0.03, 0.03, 0.05]
    elements["l"].coord.b.value = [0.2, 0.0, 0.5]
    elements["m"].coord.half_height.value = 0.2
    elements["i"].coord.b.param.value = [0.01]
    elements["j"].coord.direction.value = [1.0, 0.0, 0.0]  # the constrained spring's normal
    elements["p"].coord.goal.value = [0.0, 0.3, 0.0]
    elements["t"].max_force.value = 1.5
    data = vmc.config.load(experiment.save(tmp_path / "tuned.yaml")).to_dict()
    a = data["controller"]["elements"]["a"]
    assert a["stiffness"] == 1.5 and a["coordinate"]["difference"][0] == [0.2, 0.0, 0.3]
    assert data["coordinates"]["tip"]["point"] == {"s": 0.9, "offset": [0.0, 0.02, 0.0]}
    elements = data["controller"]["elements"]
    assert elements["b"]["coordinate"]["projection"]["direction"] == [0.0, 1.0, 1.0]
    assert elements["f"]["coordinate"]["plane_distance"]["origin"] == [0.0, 0.0, 0.25]
    assert elements["g"]["coordinate"]["sphere_distance"]["radius"] == 0.06
    assert elements["k"]["coordinate"]["box_distance"]["half_sizes"] == [0.03, 0.03, 0.05]
    assert elements["l"]["coordinate"]["capsule_distance"]["b"] == [0.2, 0.0, 0.5]
    assert elements["m"]["coordinate"]["cylinder_distance"]["half_height"] == 0.2
    assert elements["i"]["coordinate"]["difference"][1]["ref"]["value"] == [0.01]
    assert elements["j"]["normal"] == [1.0, 0.0, 0.0]
    assert elements["p"]["coordinate"]["orientation_error"]["goal"] == [0.0, 0.3, 0.0]
    assert elements["t"]["max_force"] == 1.5


def test_numbers_read_as_numbers(tmp_path):
    path = tmp_path / "numbers.yaml"
    path.write_text("a: 1e-3\nb: -2.5E+2\nc: .5\nd: 3\ne: '1e-3'\nf: [1e4, 2]\ng: true\n")
    data = files.read(path)
    expected = {"a": 0.001, "b": -250.0, "c": 0.5, "d": 3, "e": "1e-3", "f": [1e4, 2], "g": True}
    assert data == expected
    assert isinstance(data["a"], float) and isinstance(data["d"], int)


def broken(change):
    spec = copy.deepcopy(files.read(CONFIGS / "finger.yaml"))
    change(spec)
    return spec


def element(spec, **values):
    spec["controller"]["elements"]["press"].update(values)


MISTAKES = {
    "section": (lambda s: s.update(extras={}), ValueError, "unknown key 'extras'"),
    "setting": (lambda s: s["experiment"].update(duraton=1.0), ValueError, "'duraton'"),
    "type": (lambda s: element(s, type="linear_sprng"), KeyError, "no component named"),
    "gain": (lambda s: element(s, stifness=1.0), TypeError, "controller.elements.press"),
    "coordinate": (
        lambda s: s["controller"]["elements"]["press"].pop("coordinate"),
        ValueError,
        "press: linear_spring needs the `coordinate`",
    ),
    "robot": (
        lambda s: s["controller"]["elements"]["gravity"].update(coordinate="tip"),
        ValueError,
        "acts on the robot",
    ),
    "name": (lambda s: element(s, coordinate="toe"), KeyError, "no coordinate or state named"),
    "kind": (lambda s: element(s, coordinate={"pont": "tip"}), KeyError, "no coordinate named"),
    "plant": (lambda s: s["experiment"].update(plant={"type": "mujoco"}), ValueError, "mujoco"),
    "run": (lambda s: s["experiment"].update(run={"folde": "logs"}), ValueError, "'folde'"),
    "record": (
        lambda s: s["experiment"].update(run={"record": ["energies"]}),
        ValueError,
        r"experiment\.run\.record: \['energies'\]",
    ),
    "live": (
        lambda s: s["experiment"].update(schedule=[{"param": "ctrl.press.s", "points": [[0, 1]]}]),
        KeyError,
        "runtime",
    ),
    "swap": (
        lambda s: s["experiment"].update(schedule=[{"swap": "nope", "at": 1.0, "duration": 0.1}]),
        KeyError,
        "no controller named 'nope'",
    ),
    "template": (lambda s: s.update(robot="adapt.fingr"), KeyError, "no robot named"),
}


@pytest.mark.parametrize("mistake", MISTAKES)
def test_mistakes_say_where_they_are(mistake):
    change, error, message = MISTAKES[mistake]
    with pytest.raises(error, match=message):
        vmc.config.load(broken(change))


def test_a_simulation_needs_its_rate_and_duration():
    experiment = vmc.config.load(broken(lambda s: s["experiment"].pop("rate")))
    with pytest.raises(ValueError, match="rate"):
        experiment.run()


def test_a_configuration_written_in_python_saves_too(tmp_path):
    spec = files.read(CONFIGS / "finger.yaml")
    spec["controller"]["elements"]["press"]["coordinate"] = {
        "difference": ["tip", np.array([0.0, 0.05, 0.065])]
    }
    spec["experiment"]["plant"]["q0"] = (0.8, 0.8)
    saved = vmc.config.load(spec).save(tmp_path / "finger.yaml")
    assert files.read(saved) == files.read(CONFIGS / "finger.yaml")


def finger_file(tmp_path, **run):
    """The finger's file, 0.1 s long, in a folder of its own, with ``run`` settings."""
    spec = files.read(CONFIGS / "finger.yaml")
    spec["experiment"].update(duration=0.1, run=run)
    path = tmp_path / "configs" / "finger.yaml"
    path.parent.mkdir()
    return files.write(spec, path)


def test_a_run_is_saved_beside_its_file_with_the_configuration_it_started_from(
    tmp_path, monkeypatch
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    path = finger_file(tmp_path, name="first", folder="logs", record=["energy"])
    experiment = vmc.config.load(path)
    assert experiment.log_path() == path.parent / "logs" / "first.npz"
    log = experiment.run()

    saved = path.parent / "logs" / "first.npz"
    assert saved.exists() and list(elsewhere.iterdir()) == []  # relative to the file, not here
    loaded = vmc.sim.RunLog.load(saved)
    assert "energy/stored" in loaded.arrays() and "element/ctrl.press/force" not in loaded.arrays()
    for name, values in log.arrays().items():
        np.testing.assert_array_equal(loaded.arrays()[name], values, err_msg=name)

    # The log holds the configuration it started from; run again, it gives the run back.
    text = tmp_path / "from-the-log.yaml"
    text.write_text(loaded.meta["configuration"])
    assert vmc.config.load(text).to_dict() == experiment.to_dict()
    spec = files.read(text)
    del spec["experiment"]["run"]
    again = vmc.config.load(spec).run().arrays()
    recorded = {name for name in log.arrays() if "/" in name}  # what `record` added
    assert (
        set(log.arrays()) - set(again)
        == recorded
        == {"energy/stored", "energy/kinetic", "power/port", "power/dissipation", "power/source"}
    )
    for name, values in again.items():
        np.testing.assert_array_equal(loaded.arrays()[name], values, err_msg=name)

    copy_ = vmc.config.load(experiment.save(tmp_path / "copy.yaml"))
    assert copy_.to_dict() == experiment.to_dict() and copy_.run_settings == experiment.run_settings


def test_a_name_already_taken_stops_the_run_before_it_starts(tmp_path):
    experiment = vmc.config.load(finger_file(tmp_path, name="first"))
    experiment.run()
    saved = experiment.log_path()
    kept, end = saved.read_bytes(), experiment.plant.t
    assert end > 0.0
    with pytest.raises(FileExistsError, match="give the run another name"):
        experiment.run()
    assert saved.read_bytes() == kept
    assert experiment.plant.t == end  # the refused run did not start: the plant was not reset


def test_overwrite_replaces_a_log_of_the_same_name(tmp_path):
    path = finger_file(tmp_path, name="again", overwrite=True)
    experiment = vmc.config.load(path)
    experiment.run()
    experiment.mechanism.components["press"].stiffness.value = 150.0
    experiment.run()
    saved = vmc.sim.RunLog.load(experiment.log_path())
    configuration = saved.meta["configuration"]
    assert re.search(r"stiffness: 150(\.0)?\b", configuration)
    folder = experiment.log_path().parent  # the file's own: the log and no temporary file
    assert sorted(p.name for p in folder.iterdir()) == ["again.npz", "finger.yaml"]


def test_a_log_without_a_name_is_named_by_its_start_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = files.read(CONFIGS / "finger.yaml")
    spec["experiment"].update(duration=0.05, run={"folder": "logs"})  # a dict: relative to here
    vmc.config.load(spec).run()
    (saved,) = (tmp_path / "logs").iterdir()
    assert re.fullmatch(r"run-\d{8}-\d{6}\.npz", saved.name)


def test_a_name_may_carry_its_suffix(tmp_path):
    experiment = vmc.config.load(finger_file(tmp_path, name="mine.npz"))
    assert experiment.log_path().name == "mine.npz"


def test_without_run_settings_nothing_is_saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = files.read(CONFIGS / "finger.yaml")
    spec["experiment"]["duration"] = 0.05
    experiment = vmc.config.load(spec)
    log = experiment.run()
    assert experiment.log_path() is None and list(tmp_path.iterdir()) == []
    assert "configuration" in log.meta  # the log still says what it was
