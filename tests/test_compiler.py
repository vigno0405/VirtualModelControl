"""compile(): power balance, the fast motor path, live Params and error cases."""

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.core import constants
from virtualmodelcontrol.robots import helyx

rng = np.random.default_rng(8)


def rich_system():
    """Every component kind, gravity compensation and a virtual mass tied to the tip."""
    arm = helyx.arm("145-290-290")
    ctrl = vmc.Mechanism("ctrl")
    tip, mid = arm.point(s=1.0), arm.point(s=0.4)
    ctrl.add("drag", vmc.LinearSpring(tip - vmc.Ref("goal", 3, value=[0.05, 0.0, 0.6]), 40.0))
    ctrl.add("cart", vmc.TanhSpring(vmc.Projection(mid - [0.0, 0.0, 0.3], [1, 1, 0]), 80.0, 0.4))
    ctrl.add("push", vmc.GaussianSpring(mid - [0.05, 0.02, 0.25], 30.0, 0.05))
    ctrl.add("damp", vmc.LinearDamper(mid, 1.2))
    ctrl.add("gravity", vmc.GravityCompensation(arm))
    z = ctrl.add_state("follower", dim=3, unit="m", initial=[0.0, 0.0, 0.7])
    ctrl.add("mass", vmc.Inertance(z, 0.05))
    ctrl.add("tether", vmc.LinearSpring(tip - z, 25.0))
    ctrl.add("tether_d", vmc.LinearDamper(tip - z, 0.8))
    ctrl.add("drag_z", vmc.TanhDamper(z, 0.5, 0.2))
    return arm, ctrl, vmc.VirtualMechanismSystem(arm, ctrl)


def test_power_balance_holds_exactly():
    _, _, system = rich_system()
    law = vmc.compile(system)
    p = law.live_values()
    for _ in range(10):
        q, v = rng.uniform(-0.02, 0.02, 9), rng.uniform(-0.1, 0.1, 9)
        z = np.concatenate(
            [rng.uniform(-0.1, 0.1, 3) + np.array([0, 0, 0.7]), rng.uniform(-0.2, 0.2, 3)]
        )
        zdot = np.array(law.law(q, v, z, p, 0.0)[1]).ravel()
        h = 1e-6

        def E(sign, q=q, v=v, z=z, zdot=zdot, h=h):
            V, T = law.energy(q + sign * h * v, v, z + sign * h * zdot, p, 0.0)
            return float(V) + float(T)

        dEdt = (E(1) - E(-1)) / (2 * h)
        port, diss, src = (float(x) for x in law.power(q, v, z, p, 0.0))
        assert diss <= 0.0
        assert abs(dEdt - (-port + diss + src)) < 1e-6 * max(1.0, abs(port), abs(src))


def test_fast_path_equals_the_configuration_law():
    arm, _, system = rich_system()
    law = vmc.compile(system)
    p, p_act = law.live_values(), constants(arm.actuation.params)
    for _ in range(5):
        theta, theta_dot = rng.uniform(-2, 2, 9), rng.uniform(-1, 1, 9)
        z = np.concatenate([[0.0, 0.0, 0.7], np.zeros(3)])
        out = np.array(law.fast(np.concatenate([theta, theta_dot, z, p, [0.0]]))).ravel()
        q = arm.actuation.config_from_motors(ca.DM(theta), p_act)
        v = arm.actuation.config_from_motors(ca.DM(theta_dot), p_act)
        u, zdot = (np.array(x).ravel() for x in law.law(q, v, z, p, 0.0))
        np.testing.assert_allclose(out, np.concatenate([u, zdot]), atol=1e-12)


def test_set_returns_the_exact_energy_jump():
    _, _, system = rich_system()
    law = vmc.compile(system)
    controller = vmc.VMCController(law)
    theta = rng.uniform(-1, 1, 9)
    controller.step(0.0, vmc.Signals(0.0, motor_position=theta, motor_velocity=np.zeros(9)))
    before = controller.energy()
    jump = controller.set({"ctrl.drag.stiffness": 80.0, "ctrl.drag.goal": [0.0, 0.05, 0.5]})
    assert jump == pytest.approx(controller.energy() - before, abs=1e-15)
    assert jump != 0.0


def test_frozen_params_need_runtime_to_change_while_running():
    _, _, system = rich_system()
    controller = vmc.VMCController(vmc.compile(system))
    with pytest.raises(KeyError, match="runtime"):
        controller.set({"ctrl.drag.s": 0.5})  # attachment points are episode Params


def test_runtime_patterns_make_frozen_params_live():
    arm = helyx.arm()
    ctrl = vmc.Mechanism("ctrl")
    ctrl.add("k", vmc.LinearSpring(arm.point(s=1.0) - [0.0, 0.0, 0.6], 30.0))
    meas = vmc.Signals(0.0, motor_position=rng.uniform(-1, 1, 9), motor_velocity=np.zeros(9))
    live = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl), runtime=["*.s"]))
    live.set({"ctrl.k.s": 0.5})
    ctrl.components["k"].coord.a.s.value = 0.5  # the same change, then compile again
    frozen = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    np.testing.assert_allclose(
        live.step(0.0, meas)["motor_torque"], frozen.step(0.0, meas)["motor_torque"], atol=1e-15
    )


def test_system_params_and_errors():
    arm, ctrl, system = rich_system()
    names = list(system.params)
    assert "arm.gravity" in names and "ctrl.gravity.gravity" not in names  # shared: robot's name
    assert "ctrl.drag.stiffness" in names and names.index("arm.seg1.L0") == 0
    with pytest.raises(ValueError, match="kinematic model"):
        vmc.VirtualMechanismSystem(vmc.Mechanism("no_model"), ctrl)
    lonely = vmc.Mechanism("ctrl")
    z = lonely.add_state("z")
    lonely.add("k", vmc.LinearSpring(z, 1.0))
    with pytest.raises(ValueError, match="inertance"):
        vmc.compile(vmc.VirtualMechanismSystem(arm, lonely))
    heavy = vmc.Mechanism("ctrl")
    heavy.add("m", vmc.Inertance(arm.point(s=1.0), 0.1))
    with pytest.raises(ValueError, match="states only"):
        vmc.compile(vmc.VirtualMechanismSystem(arm, heavy))
