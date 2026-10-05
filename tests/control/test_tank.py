"""The energy tank: a change of a running controller, paid for with exact energy."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from small_plan import mass_spring, tanh_mass
from virtualmodelcontrol import optimization as opt
from virtualmodelcontrol.control import Tank
from virtualmodelcontrol.robots import helyx

K, G = "ctrl.spring.stiffness", "ctrl.spring.goal"
X, GOAL, STIFFNESS = 0.3, 1.0, 4.0
DEFLECTION2 = (X - GOAL) ** 2  # the spring's deflection squared at the controller's state


def running(system=None):
    """The controller, after one step at x = 0.3."""
    controller = vmc.VMCController(vmc.compile(system or mass_spring()[0]))
    controller.reset(0.0)
    controller.step(0.0, vmc.Signals(0.0, motor_position=[X], motor_velocity=[0.0]))
    return controller


def test_a_stiffness_step_is_applied_as_far_as_the_tank_pays():
    controller = running()
    tank = Tank(controller, level=0.5)
    full = 0.5 * (10.0 - STIFFNESS) * DEFLECTION2  # [J]
    jump = tank.set({K: 10.0})
    assert jump == pytest.approx(0.5, rel=1e-9) and tank.level == pytest.approx(0.0, abs=1e-12)
    assert tank.fraction == pytest.approx(0.5 / full, rel=1e-9)
    assert controller.live_params()[K] == pytest.approx(STIFFNESS + tank.fraction * 6.0)


def test_a_step_that_fits_is_applied_whole():
    tank = Tank(running(), level=5.0)
    jump = tank.set(**{K: 10.0})
    assert tank.fraction == 1.0 and jump == pytest.approx(0.5 * 6.0 * DEFLECTION2)
    assert tank.level == pytest.approx(5.0 - jump)


def test_a_step_that_releases_energy_is_applied_whole_with_an_empty_tank_and_refills_it():
    controller = running()
    tank = Tank(controller)  # empty
    jump = tank.set({K: 2.0})
    assert tank.fraction == 1.0 and jump == pytest.approx(-0.5 * 2.0 * DEFLECTION2)
    assert tank.level == pytest.approx(-jump) and controller.live_params()[K] == 2.0
    capped = Tank(running(), capacity=0.3)
    capped.set({K: 2.0})
    assert capped.level == 0.3  # the rest is dissipated


def test_a_reference_step_costs_a_quadratic_and_is_still_bounded_exactly():
    # the goal moves from 1 to 2: the jump is 2 a^2 + 2.8 a, at the fraction a of the step
    tank = Tank(running(), level=1.0)
    tank.set({G: [2.0]})
    assert tank.fraction == pytest.approx((-2.8 + np.sqrt(2.8**2 + 8.0)) / 4.0, rel=1e-9)
    assert tank.level == pytest.approx(0.0, abs=1e-9)


def test_the_tank_never_goes_negative_and_keeps_the_books():
    rng = np.random.default_rng(3)
    controller = running()
    tank = Tank(controller, level=0.4)
    total, before = 0.0, controller.energy()
    for _ in range(40):
        jump = tank.set({K: rng.uniform(0.5, 30.0), G: [rng.uniform(-1.0, 2.0)]})
        total += jump
        assert tank.level >= 0.0
    assert tank.level == pytest.approx(0.4 - total, abs=1e-9)
    assert controller.energy() - before == pytest.approx(total, abs=1e-9)  # at the same state


def test_before_the_first_step_the_whole_change_is_applied():
    controller = vmc.VMCController(vmc.compile(mass_spring()[0]))
    tank = Tank(controller)
    assert tank.set({K: 100.0}) == 0.0 and tank.fraction == 1.0
    assert controller.live_params()[K] == 100.0


def test_a_param_that_is_not_live_is_refused_with_the_controllers_message():
    with pytest.raises(KeyError, match="not a live Param"):
        Tank(running()).set({"robot.mass.inertance": 2.0})


def test_jump_tells_the_energy_jump_of_set_without_changing_anything():
    controller = running()
    jump = controller.jump({K: 10.0})
    assert controller.live_params()[K] == STIFFNESS
    assert controller.set({K: 10.0}) == pytest.approx(jump)


def test_an_optimizers_result_passes_through_the_tank():
    system, x = tanh_mass()
    problem = opt.Problem(system)
    problem.add(opt.Collocation([0.0], 4.0, 41))
    problem.free(K)
    problem.add(opt.Effort(1e-2))
    problem.add(opt.Cost(x - 1.0, name="reach"))
    result = problem.solve()
    controller = running(system)
    start = controller.live_params()[K]
    tank = Tank(controller)  # empty: a stiffer spring cannot be paid for
    result.apply(tank)
    assert result.params[K] > start and tank.fraction == pytest.approx(0.0, abs=1e-12)
    assert controller.live_params()[K] == pytest.approx(start)


def soft_arm_run(tank_args=None, T=0.1, dt=1 / 5000):
    """The soft arm held by a spring and a damper, run with the tank in place of the controller."""
    arm = helyx.add_dynamics(helyx.arm("145-145-145"))
    tip = arm.point(s=1.0)
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("reach", vmc.LinearSpring(tip - [0.08, 0.0, 0.40], 300.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    tank = Tank(controller, **(tank_args or {}))
    log = vmc.sim.run(vmc.sim.ModelPlant(arm), tank, vmc.sim.SimClock(dt), T=T, record=["energy"])
    return tank, log


def test_a_tank_in_the_loop_fills_with_what_the_dampers_take():
    tank, log = soft_arm_run()
    taken = vmc.sim.energy_balance(log)["dissipated"][-1]
    assert taken > 0.1
    assert tank.level == pytest.approx(taken, rel=0.02)


def test_the_tank_in_the_loop_never_holds_more_than_its_capacity():
    tank, log = soft_arm_run({"capacity": 0.1})
    assert vmc.sim.energy_balance(log)["dissipated"][-1] > 0.1
    assert tank.level == 0.1


def test_what_the_dampers_gave_pays_for_a_stiffer_spring():
    tank, _ = soft_arm_run()
    filled, stiffer = tank.level, {"ctrl.reach.stiffness": 3000.0}
    assert tank.controller.jump(stiffer) > filled  # the whole step costs more than the tank holds
    jump = tank.set(stiffer)
    assert tank.fraction < 1.0 and jump == pytest.approx(filled, rel=1e-6)
    assert tank.level == pytest.approx(0.0, abs=1e-9)


def test_a_reset_keeps_the_level_and_the_first_step_after_it_adds_nothing():
    tank, _ = soft_arm_run()
    level = tank.level
    meas = vmc.Signals(0.0, motor_position=np.zeros(9), motor_velocity=np.full(9, 0.5))
    tank.reset(0.0, meas)
    assert tank.level == level
    tank.step(0.0, meas)
    assert tank.level == level
