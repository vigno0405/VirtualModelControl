"""The energy balance of a logged run."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

DT = 1 / 5000


def soft_arm():
    arm = helyx.add_dynamics(helyx.arm("145-145-145"))
    tip = arm.point(s=1.0)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(tip - [0.08, 0.0, 0.40], 300.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    return arm, vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))


def run(steps=1000, jump_at=None, stiffness=None):
    """A run of the soft arm that logs its energies; optionally a stiffness set before a step.
    Returns the log and the energy jump ``set`` reported."""
    arm, controller = soft_arm()
    plant, log, jump = vmc.sim.ModelPlant(arm), vmc.sim.RunLog(), 0.0
    controller.reset(plant.t, plant.read())
    for k in range(steps):
        if k == jump_at:
            jump = controller.set({"ctrl.reach.stiffness": stiffness})
        meas = plant.read()
        cmd = controller.step(plant.t, meas)
        plant.write(cmd)
        balance = controller.balance()
        log.step(
            t=plant.t,
            **{f"energy/{name}": balance[name] for name in ("stored", "kinetic")},
            **{f"power/{name}": balance[name] for name in ("port", "dissipation", "source")},
        )
        plant.advance(DT)
    return log, jump


def test_the_balance_closes_on_a_run_that_changes_no_parameter():
    log, _ = run()
    b = vmc.sim.energy_balance(log)
    assert b["dissipated"][-1] > 1e-3 and b["given"][-1] > 1e-3  # a run with something to count
    assert np.abs(b["injected"]).max() < 0.01 * b["dissipated"][-1]
    assert (np.diff(b["dissipated"]) >= -1e-12).all()  # dampers only take
    assert np.abs(b["supplied"]).max() < 1e-6  # gravity compensation does no net work here
    assert b["margin"].min() > 0.0 and b["margin"][0] == pytest.approx(b["energy"][0])


def test_a_change_of_stiffness_shows_up_as_injected_energy_of_the_size_of_its_jump():
    log, jump = run(jump_at=500, stiffness=1500.0)
    b = vmc.sim.energy_balance(log)
    assert jump > 1e-3
    before, after = b["injected"][499], b["injected"][-1]
    assert abs(before) < 0.01 * jump  # the error of the steps is small beside the jump
    assert after - before == pytest.approx(jump, rel=0.02)
    assert log.arrays()["t"].ravel()[0] == b["t"][0]


def test_the_margin_is_what_the_controller_can_still_give_and_goes_negative_when_it_cheats():
    honest, _ = run(steps=2000)
    cheating, _ = run(steps=2000, jump_at=300, stiffness=30000.0)
    margin = [vmc.sim.energy_balance(log)["margin"] for log in (honest, cheating)]
    assert margin[0].min() > 0.0
    assert margin[1].min() < 0.0  # it gave the robot more than it ever stored


def test_a_log_without_energies_is_refused():
    arm, controller = soft_arm()
    log = vmc.sim.run(vmc.sim.ModelPlant(arm), controller, vmc.sim.SimClock(DT), T=0.01)
    with pytest.raises(ValueError, match="record="):
        vmc.sim.energy_balance(log)


def test_a_step_that_computed_nothing_counts_as_zero_power():
    log, _ = run(steps=200)
    for name, rows in log.rows.items():
        if name != "t":  # one step the controller did not compute: the time is still known
            rows[100] = np.full_like(rows[100], np.nan)
    b = vmc.sim.energy_balance(log)
    assert np.isfinite(b["given"]).all() and np.isfinite(b["dissipated"]).all()
    assert np.isnan(b["energy"][100]) and np.isfinite(b["energy"][99])


def test_sources_and_virtual_masses_are_counted():
    robot = vmc.Mechanism("robot", model=vmc.models.JointSpace(1, unit="m"))
    robot.add("mass", vmc.Inertance(robot.joint(0), 1.0))
    robot.add("friction", vmc.LinearDamper(robot.joint(0), 1.0))
    ctrl = vmc.Mechanism("ctrl")
    phi = ctrl.add_state("phi", unit="m")
    ctrl.add("virtual_mass", vmc.Inertance(phi, 0.5))
    ctrl.add("tie", vmc.LinearSpring(robot.joint(0) - phi, 20.0))
    ctrl.add("tie_damper", vmc.LinearDamper(robot.joint(0) - phi, 2.0))
    ctrl.add("pull", vmc.LinearSpring(phi - 1.0, 10.0))
    ctrl.add("drag", vmc.LinearDamper(phi, 1.0))
    ctrl.add("push", vmc.ForceSource(phi, 3.0))  # a source that does net work
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    log = vmc.sim.run(
        vmc.sim.ModelPlant(robot), controller, vmc.sim.SimClock(1e-3), T=3.0, record="energy"
    )
    b = vmc.sim.energy_balance(log)
    assert b["supplied"][-1] > 1.0  # the source gave the controller energy
    assert log.arrays()["energy/kinetic"].max() > 0.1  # and the virtual mass holds some
    assert np.abs(b["injected"]).max() < 0.01 * b["supplied"][-1]
    # what it can still give is what it holds and what its dampers took, less what came in
    np.testing.assert_allclose(b["margin"], b["energy"] + b["dissipated"] - b["injected"])
