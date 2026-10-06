"""The law in a closed loop: a fingertip presses a table with a force it is asked for."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.adaptation import ForceTracking
from virtualmodelcontrol.robots import adapt

K_TABLE = 1e4
WANTED = np.array([0.0, 0.0, 2.0])


@pytest.fixture(scope="module")
def compiled():
    finger = adapt.add_dynamics(adapt.finger())
    tip = finger.point("tip")
    gap = vmc.PlaneDistance(tip, normal=[0, 0, -1], origin=[0, 0, 0.06])
    finger.add("table", vmc.ContactSpring(gap, K_TABLE))
    finger.add("cushion", vmc.ContactDamper(gap, 5.0))
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("press", vmc.LinearSpring(tip - vmc.Ref("goal", 3, value=[0, 0.05, 0.06]), 100.0))
    ctrl.add("damp", vmc.LinearDamper(tip, 1.0))
    ctrl.add("limits", adapt.joint_limit_spring(finger))
    ctrl.add("gravity", vmc.GravityCompensation(finger))
    return finger, vmc.compile(vmc.VirtualMechanismSystem(finger, ctrl))


def press(finger, controller, steps):
    law = ForceTracking(controller, "tip", "ctrl.press.goal", normal=[0, 0, 1])
    plant = vmc.sim.ModelPlant(finger, q0=[0.8, 0.8], max_step=1e-4)
    kin = vmc.Kinematics(finger)
    forces, levels = [], []
    for step in range(steps):
        plant.write(controller.step(plant.t, plant.read()))
        below = kin.position(plant.q, "tip")[2] - 0.06
        f = np.array([0.0, 0.0, K_TABLE * max(0.0, below)])
        if step > 100:
            law.step(controller, f, WANTED)
        forces.append(f[2])
        levels.append(getattr(controller, "level", 0.0))
        plant.advance(1 / 500)
    return np.array(forces), np.array(levels)


def test_the_table_force_reaches_the_wanted_one_and_stays(compiled):
    finger, system = compiled
    forces, _ = press(finger, vmc.VMCController(system), 1500)
    assert forces[-300:].mean() == pytest.approx(WANTED[2], abs=0.02)
    assert np.ptp(forces[-300:]) < 0.1


def test_an_empty_tank_stalls_and_a_funded_one_reaches_it(compiled):
    finger, system = compiled
    stalled, level = press(finger, vmc.control.Tank(vmc.VMCController(system), level=0.0), 1500)
    assert stalled[-1] < 0.8 * WANTED[2] and level.min() >= -1e-9
    funded, level = press(finger, vmc.control.Tank(vmc.VMCController(system), level=0.05), 1500)
    assert funded[-300:].mean() == pytest.approx(WANTED[2], abs=0.02) and level.min() >= -1e-9
