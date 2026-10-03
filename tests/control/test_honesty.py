"""Honesty models: the same machinery on a rigid arm, a virtual flywheel and a floating body."""

import casadi as ca
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.core import ParamSet
from virtualmodelcontrol.models import SerialChain

rng = np.random.default_rng(9)


def signals(q, v):
    return vmc.Signals(0.0, motor_position=q, motor_velocity=v)


# --- 2-link planar arm (product of exponentials, direct drive) -----------------------------


def test_two_link_arm_torque_is_jacobian_transpose_force():
    L1, L2, k, c = 0.3, 0.2, 40.0, 2.0
    z = [0.0, 0.0, 1.0]
    model = SerialChain(
        ["revolute", "revolute"],
        axes=[z, z],
        points=[[0, 0, 0], [L1, 0, 0]],
        sites={"tip": (2, [L1 + L2, 0, 0])},
    )
    arm, ctrl = vmc.Mechanism("arm", model=model), vmc.Mechanism("ctrl")
    goal = np.array([0.25, 0.2, 0.0])
    ctrl.add("k", vmc.LinearSpring(arm.point("tip") - goal, k))
    ctrl.add("c", vmc.LinearDamper(arm.point("tip"), c))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
    for q, v in zip(rng.uniform(-3, 3, (10, 2)), rng.uniform(-1, 1, (10, 2)), strict=True):
        s1, c1, s12, c12 = np.sin(q[0]), np.cos(q[0]), np.sin(q.sum()), np.cos(q.sum())
        tip = np.array([L1 * c1 + L2 * c12, L1 * s1 + L2 * s12, 0.0])
        J = np.array([[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12], [0, 0]])
        f = k * (goal - tip) - c * J @ v
        u = controller.step(0.0, signals(q, v))["motor_torque"]
        np.testing.assert_allclose(u, J.T @ f, atol=1e-12)


# --- virtual flywheel coupled to two cranks (a controller with its own state) --------------


class Cranks:
    """Two independent crank angles [rad]; no frames needed."""

    q_unit = "rad"

    def __init__(self):
        self.space, self.params, self.sites = vmc.Euclidean(2), ParamSet(), ()


def test_flywheel_follows_the_coupled_crank_law():
    K, C, J_v, b_v, delta = 1.0, 0.05, 0.1, 0.1, np.pi
    robot, ctrl = vmc.Mechanism("turtle", model=Cranks()), vmc.Mechanism("ctrl")
    phi = ctrl.add_state("phi", unit="rad")
    ctrl.add("flywheel", vmc.Inertance(phi, J_v))
    left = robot.joint(0) - phi
    right = robot.joint(1) - (phi - vmc.Ref("delta", 1, value=delta, unit="rad"))
    for name, e in (("left", left), ("right", right)):
        ctrl.add(f"k_{name}", vmc.LinearSpring(e, K))
        ctrl.add(f"c_{name}", vmc.LinearDamper(e, C))
    ctrl.add("drag", vmc.LinearDamper(phi, b_v))
    controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))

    p, pd, dt = 0.0, 0.0, 0.01  # the same law written out by hand, semi-implicit Euler
    for k in range(200):
        q, v = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
        e = q - np.array([p, p - delta])
        ed = v - pd
        u = controller.step(k * dt, signals(q, v))["motor_torque"]
        np.testing.assert_allclose(u, -K * e - C * ed, atol=1e-12)
        elapsed = 0.0 if k == 0 else dt  # integrate over the time since the last step
        pd += elapsed * (np.sum(K * e + C * ed) - b_v * pd) / J_v
        p += elapsed * pd
    np.testing.assert_allclose(controller.z, [p, pd], atol=1e-10)


# --- planar floating body, nq = 4 (x, y, cos θ, sin θ) ≠ nv = 3 ------------------------------


class PlanarBody:
    """A rigid body in the plane; site ``marker`` sits at ``offset`` in the body frame."""

    q_unit = ""

    def __init__(self):
        self.space = vmc.Product(vmc.Euclidean(2), vmc.SO2())
        self.params = ParamSet([vmc.Param("offset", [0.1, 0.05, 0.0], unit="m")])
        self.sites = ("marker",)

    def frame(self, q, at, p):
        c, s = q[2], q[3]
        R = ca.vertcat(ca.horzcat(c, -s, 0), ca.horzcat(s, c, 0), ca.horzcat(0, 0, 1))
        return R, ca.vertcat(q[0], q[1], 0) + ca.mtimes(R, p["offset"])


def test_planar_body_on_a_manifold():
    body, ctrl = vmc.Mechanism("body", model=PlanarBody()), vmc.Mechanism("ctrl")
    ctrl.add("k", vmc.LinearSpring(body.point("marker") - [0.3, -0.2, 0.0], 20.0))
    ctrl.add("c", vmc.LinearDamper(body.point("marker"), 0.5))
    law = vmc.compile(vmc.VirtualMechanismSystem(body, ctrl))
    space, p = body.space, law.live_values()

    # Storage power: τᵀv = −dV/dt along the manifold, for random states.
    for _ in range(10):
        q = space.integrate(space.neutral(), rng.uniform(-1, 1, 3))
        v, h = rng.normal(size=3), 1e-6
        V = lambda x, v=v: float(law.energy(x, v, [], p, 0.0)[0])  # noqa: E731
        dVdt = (V(space.integrate(q, h * v)) - V(space.integrate(q, -h * v))) / (2 * h)
        port, diss, _ = (float(x) for x in law.power(q, v, [], p, 0.0))
        assert abs(port - (-dVdt + diss)) < 1e-6

    # Closed loop with unit mass and inertia: energy decays, q stays on the manifold.
    controller, q, v, dt = (
        vmc.VMCController(law),
        space.integrate(space.neutral(), [0, 0, 2.0]),
        np.zeros(3),
        1e-3,
    )
    energies = []
    for k in range(3000):
        tau = controller.step(k * dt, signals(q, v))["motor_torque"]
        v = v + dt * tau
        q = space.integrate(q, dt * v)
        energies.append(controller.energy() + 0.5 * v @ v)
    assert abs(np.hypot(q[2], q[3]) - 1.0) < 1e-12
    assert energies[-1] < 0.5 * energies[0]
