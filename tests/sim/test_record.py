"""What a run records: the law's torque, extras on request, and what the run was."""

import json
from datetime import datetime

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.control.output import apply
from virtualmodelcontrol.robots import adapt, helyx

DT = 1 / 330
ELEMENTS = ("ctrl.reach", "ctrl.damp", "ctrl.gravity")
ROBOT = ("stiffness", "damping", "gravity")  # the arm's own springs, dampers and gravity
BASIC = {"t", "motor_torque", "law_torque", "motor_position", "motor_velocity", "q", "v"}


def soft_arm(output=None):
    arm = helyx.add_dynamics(helyx.arm("145-145-145"))
    tip = arm.point(s=1.0)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(tip - vmc.Ref("goal", value=[0.05, 0.0, 0.40]), 300.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    law = vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))
    return arm, vmc.VMCController(law, output=output)


def test_a_run_logs_what_it_measured_and_what_it_sent():
    arm, controller = soft_arm()
    log = vmc.sim.run(vmc.sim.ModelPlant(arm), controller, vmc.sim.SimClock(DT), T=0.1)
    rows = log.arrays()
    assert set(rows) == BASIC
    np.testing.assert_array_equal(rows["law_torque"], rows["motor_torque"])  # no output stage


def test_the_law_torque_is_logged_before_the_output_stages():
    arm, controller = soft_arm(output=helyx.output_stage())
    log = vmc.sim.run(vmc.sim.ModelPlant(arm), controller, vmc.sim.SimClock(DT), T=0.1)
    rows = log.arrays()
    assert np.abs(rows["motor_torque"] - rows["law_torque"]).max() > 1e-3
    first = vmc.Signals(
        0.0, motor_position=rows["motor_position"][0], motor_velocity=rows["motor_velocity"][0]
    )
    sent = apply(controller.output, rows["law_torque"][0], first)
    np.testing.assert_allclose(sent, rows["motor_torque"][0], atol=1e-15)


def test_record_adds_the_live_params_the_elements_and_the_energies():
    arm, controller = soft_arm()
    clock = vmc.sim.SimClock(DT)
    log = vmc.sim.run(vmc.sim.ModelPlant(arm), controller, clock, T=0.2, record=vmc.sim.RECORDS)
    rows = log.arrays()
    live = {"ctrl.reach.stiffness": (1,), "ctrl.reach.goal": (3,), "ctrl.damp.damping": (1,)}
    names = {f"param/{name}" for name in live}
    names |= {f"element/{e}/{q}" for e in ELEMENTS for q in ("y", "ydot", "force", "torque")}
    names |= {"energy/stored", "energy/kinetic", "power/port", "power/dissipation", "power/source"}
    names |= {f"robot/{e}/{q}" for e in ROBOT for q in ("y", "ydot", "force", "torque")}
    assert set(rows) == BASIC | names
    n = len(rows["t"])
    assert all(len(values) == n for values in rows.values())

    for name, shape in live.items():  # the Params as the controller holds them, in their shapes
        assert rows[f"param/{name}"].shape == (n, *shape)
    np.testing.assert_array_equal(rows["param/ctrl.reach.goal"][0], [0.05, 0.0, 0.40])
    assert rows["element/ctrl.reach/y"].shape == (n, 3)  # the spring's coordinate: tip - goal
    assert rows["element/ctrl.reach/torque"].shape == (n, 9)  # as motor torques
    assert rows["robot/stiffness/force"].shape == (n, 9)  # the arm's own stiffness, in q


def test_a_matrix_param_is_logged_in_its_own_shape():
    arm = helyx.add_dynamics(helyx.arm("145-145-145"))
    K = np.array([[3.0, 1.0, 0.0], [0.0, 4.0, 2.0], [1.0, 0.0, 5.0]])  # not symmetric
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("hold", vmc.LinearSpring(arm.point(s=1.0) - [0.0, 0.0, 0.40], K))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    plant = vmc.sim.ModelPlant(arm)
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(DT), T=0.03, record="params")
    held = log.arrays()["param/ctrl.hold.stiffness"]
    assert held.shape == (10, 3, 3)
    np.testing.assert_array_equal(held[0], K)
    assert log.meta["params"]["ctrl.hold.stiffness"]["value"] == K.tolist()


def test_the_elements_shares_add_up_to_the_law_torque():
    arm, controller = soft_arm()
    log = vmc.sim.run(
        vmc.sim.ModelPlant(arm), controller, vmc.sim.SimClock(DT), T=0.3, record="elements"
    )
    rows = log.arrays()
    shares = sum(rows[f"element/{e}/torque"] for e in ELEMENTS)
    np.testing.assert_allclose(shares, rows["law_torque"], atol=1e-12)
    # A spring's force is its stiffness times the deflection, and its coordinate is that deflection.
    np.testing.assert_allclose(
        rows["element/ctrl.reach/force"], -300.0 * rows["element/ctrl.reach/y"], atol=1e-12
    )


def test_the_recorded_energies_are_the_controllers_own():
    arm, controller = soft_arm()
    log = vmc.sim.run(
        vmc.sim.ModelPlant(arm), controller, vmc.sim.SimClock(DT), T=0.3, record=["energy"]
    )
    rows = log.arrays()
    last = rows["energy/stored"][-1] + rows["energy/kinetic"][-1]
    assert last == pytest.approx(controller.energy(), rel=1e-12)
    assert controller.balance()["stored"] == rows["energy/stored"][-1, 0]
    port = np.sum(rows["law_torque"] * rows["motor_velocity"], axis=1)  # tau.v = u.theta_dot
    np.testing.assert_allclose(rows["power/port"].ravel(), port, atol=1e-9)
    assert rows["power/dissipation"].max() <= 1e-12  # dampers never give energy back
    assert np.all(rows["energy/kinetic"] == 0.0)  # no virtual states, no virtual masses


def test_record_takes_known_names_only():
    arm, controller = soft_arm()
    plant, clock = vmc.sim.ModelPlant(arm), vmc.sim.SimClock(DT)
    with pytest.raises(ValueError, match="record takes some of"):
        vmc.sim.run(plant, controller, clock, T=0.1, record=["energies"])
    with pytest.raises(ValueError, match="record takes some of"):
        vmc.sim.run(plant, controller, clock, T=0.1, record="Params")


class Glitchy(vmc.sim.ModelPlant):
    """A simulated arm whose velocity readings are NaN from 15 to 35 ms."""

    def read(self):
        meas = super().read()
        if 0.015 <= self.t < 0.035:
            meas.set("motor_velocity", np.full(9, np.nan))
        return meas


def test_a_step_the_guard_stops_has_zero_torque_and_no_extras():
    arm, controller = soft_arm()
    guard = vmc.sim.Guard()
    log = vmc.sim.run(
        Glitchy(arm),
        controller,
        vmc.sim.SimClock(0.01),
        T=0.06,
        guard=guard,
        record=vmc.sim.RECORDS,
    )
    rows = log.arrays()
    assert guard.trips == 2
    stopped = np.isnan(rows["motor_velocity"]).any(axis=1)
    assert stopped.tolist() == [False, False, True, True, False, False]
    np.testing.assert_array_equal(rows["motor_torque"][stopped], 0.0)
    for name in set(rows) - {"t", "motor_torque", "motor_position", "motor_velocity", "q", "v"}:
        assert np.isnan(rows[name][stopped]).all(), name  # law torque and extras: not computed
        assert np.isfinite(rows[name][~stopped]).all(), name
    assert np.isfinite(rows["t"]).all() and np.isfinite(rows["q"]).all()


def test_a_run_says_what_it_was(tmp_path):
    arm, controller = soft_arm()
    controller.set({"ctrl.reach.stiffness": 450.0})  # the controller's own value, not the Param's
    plant = vmc.sim.ModelPlant(arm)
    plant.profile = adapt.finger_hardware()  # a plant may carry its hardware profile
    log = vmc.sim.run(plant, controller, vmc.sim.SimClock(DT), T=0.05)
    meta = log.meta
    assert meta["library"] == vmc.__version__
    assert datetime.fromisoformat(meta["start"]).utcoffset() is not None  # local time, with offset
    params = meta["params"]
    assert params["ctrl.reach.stiffness"] == {"value": 450.0, "unit": "N/m"}
    assert params["ctrl.reach.goal"]["value"] == [0.05, 0.0, 0.40]
    assert params["ctrl.damp.damping"]["value"] == 5.0  # a Param of the controller, set or not
    assert "arm.seg1.L0" in params  # and every other Param of the system
    assert meta["hardware"] == adapt.finger_hardware().to_dict()
    back = vmc.sim.RunLog.load(log.save(tmp_path / "run")).meta  # and all of it can be stored
    assert back["hardware"]["rate"] == adapt.finger_hardware().rate
    assert back["params"] == json.loads(json.dumps(params)) and back["library"] == meta["library"]


def test_a_real_time_run_logs_the_same_things():
    arm, controller = soft_arm()
    plant = vmc.sim.ModelPlant(arm)
    clock = vmc.sim.WallClock(DT, now=lambda: plant.t, sleep=plant.advance)  # time moves on sleep
    log = vmc.sim.run(plant, controller, clock, T=0.1, record=["elements", "energy"])
    rows = log.arrays()
    assert BASIC | {"dt"} <= set(rows) and "element/ctrl.reach/force" in rows
    assert len(rows["t"]) == log.info["steps"] == round(0.1 / DT)
    assert log.meta["library"] == vmc.__version__ and "ctrl.reach.stiffness" in log.meta["params"]
    np.testing.assert_allclose(
        sum(rows[f"element/{e}/torque"] for e in ELEMENTS), rows["law_torque"], atol=1e-12
    )


def test_a_swap_changes_what_is_recorded_without_misaligning_the_log():
    arm, first = soft_arm()
    tip = arm.point(s=1.0)
    other = vmc.Mechanism("other")
    other.add("hold", vmc.LinearSpring(tip - [0.0, 0.0, 0.40], 100.0))
    second = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, other)))
    controller = vmc.control.ScheduledController(first, swaps=[(0.05, second, 0.02)])
    log = vmc.sim.run(
        vmc.sim.ModelPlant(arm), controller, vmc.sim.SimClock(0.01), T=0.12, record="elements"
    )
    rows = log.arrays()
    assert all(len(values) == 12 for values in rows.values())
    reach, hold = rows["element/ctrl.reach/force"], rows["element/other.hold/force"]
    assert np.isfinite(reach[:7]).all() and np.isnan(reach[7:]).all()  # until the swap is done
    assert np.isnan(hold[:7]).all() and np.isfinite(hold[7:]).all()  # and after it
