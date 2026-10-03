"""Contact: signed distances, one-sided stiff springs, contact damping, landing on a floor."""

import casadi as ca
import numpy as np

import virtualmodelcontrol as vmc
from helpers import component_function, component_params, make_context
from virtualmodelcontrol.models import SerialChain

P = vmc.Joint(slice(0, 3), unit="m")


def test_signed_distances():
    plane = component_function(
        vmc.ContactSpring(vmc.PlaneDistance(P, [0, 0, 2.0], [0, 0, 0.1]), 1.0), 3
    )
    sphere = component_function(vmc.ContactSpring(vmc.SphereDistance(P, [0, 0, 0], 0.5), 1.0), 3)
    assert abs(float(plane([0.3, -0.2, 0.4], np.zeros(3))[2]) - 0.3) < 1e-15  # normal normalized
    assert abs(float(sphere([0.0, 0.3, 0.4], np.zeros(3))[2]) - 0.0) < 1e-12


def test_contact_spring_pushes_only_in_contact():
    fn = component_function(vmc.ContactSpring(vmc.PlaneDistance(P, [0, 0, 1]), 1000.0), 3)
    assert np.all(np.array(fn([0.1, 0.2, 0.01], np.zeros(3))[0]) == 0.0)  # separated: no force
    tau = np.array(fn([0.0, 0.0, -0.002], np.zeros(3))[0]).ravel()
    np.testing.assert_allclose(tau, [0, 0, 2.0], atol=1e-12)  # 1000 N/m x 2 mm, pushing up


def test_smoothed_contact_is_close_and_differentiable():
    exact = component_function(vmc.ContactSpring(vmc.PlaneDistance(P, [0, 0, 1]), 1e5), 3)
    smooth = component_function(vmc.ContactSpring(vmc.PlaneDistance(P, [0, 0, 1]), 1e5, 1e-5), 3)
    for z in (-1e-3, -1e-4):
        a, b = float(exact([0, 0, z], np.zeros(3))[1]), float(smooth([0, 0, z], np.zeros(3))[1])
        assert abs(a - b) < 1e-3 * abs(a) + 1e-2
    spring = vmc.ContactSpring(vmc.PlaneDistance(P, [0, 0, 1]), 1e5, 1e-5)
    ctx = make_context(component_params(spring), 3)
    f = spring.force(ctx, ctx.value(spring.coord), 0)
    k = ca.Function("k", [ctx.q], [ca.jacobian(f, ctx.q)])
    assert np.all(np.isfinite(np.array(k([0, 0, 0]))))  # smooth stiffness at the surface


def test_contact_damper_acts_only_in_contact():
    fn = component_function(vmc.ContactDamper(vmc.PlaneDistance(P, [0, 0, 1]), 50.0), 3)
    assert float(fn([0, 0, 0.01], [0, 0, -1.0])[1]) == 0.0
    assert float(fn([0, 0, -0.01], [0, 0, -1.0])[1]) == 50.0  # f = −D ḋ, pushes against motion
    smooth = vmc.ContactDamper(vmc.PlaneDistance(P, [0, 0, 1]), 50.0, 1e-4)
    ctx = make_context(component_params(smooth), 3)
    v = ca.SX.sym("v", 3)
    f = smooth.force(ctx, ctx.value(smooth.coord), v[2])
    df = ca.Function("df", [ctx.q, v], [ca.jacobian(f, ca.vertcat(ctx.q, v))])
    for z in (1.0, 1e-4, -1.0):  # far outside, at the edge, deep inside: finite derivatives
        assert np.all(np.isfinite(np.array(df([0, 0, z], [0, 0, -1.0]))))


def test_a_mass_lands_on_a_stiff_floor_and_settles_at_the_right_depth():
    # a 0.1 kg mass on a vertical slider, starting 5 cm above a floor of 1e5 N/m
    chain = SerialChain(["prismatic"], [[0, 0, 1]], [[0, 0, 0]], {"foot": (1, [0, 0, 0])})
    body = vmc.Mechanism("body", model=chain)
    body.add_param(vmc.Param("gravity", [0, 0, -9.81], unit="m/s^2"))
    foot = body.point("foot")
    body.add("mass", vmc.PointMass(foot, 0.1))
    body.add("weight", vmc.Gravity(body))
    floor = vmc.PlaneDistance(foot, [0, 0, 1])
    body.add("floor", vmc.ContactSpring(floor, 1e5))
    body.add("floor_damping", vmc.ContactDamper(floor, 20.0))
    plant = vmc.sim.ModelPlant(body, q0=[0.05], max_step=1e-4)
    plant.advance(1.5)
    assert np.isfinite(plant.q[0]) and abs(plant.v[0]) < 1e-6
    np.testing.assert_allclose(-plant.q[0], 0.1 * 9.81 / 1e5, rtol=1e-6)  # m g / k
