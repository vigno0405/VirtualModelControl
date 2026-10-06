"""A rail: a joint that slides along the natural cubic spline through its waypoints."""

import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

import virtualmodelcontrol as vmc
from virtualmodelcontrol.dynamics import compile_dynamics
from virtualmodelcontrol.models import SerialChain, evaluate_frame, from_dict
from virtualmodelcontrol.testing import check_model

WAYPOINTS = np.array(
    [[0.0, 0.0, 0.0], [0.1, 0.05, 0.0], [0.2, 0.0, 0.1], [0.3, -0.05, 0.1], [0.4, 0.0, 0.0]]
)


def rail(waypoints=WAYPOINTS, site=(0.0, 0.0, 0.0)):
    return SerialChain(
        [("rail", waypoints)], axes=[None], points=[[0.0, 0.0, 0.0]], sites={"cart": (1, site)}
    )


def test_the_cart_follows_the_natural_cubic_spline_through_the_waypoints():
    kin = vmc.Kinematics(rail())
    path = CubicSpline(np.arange(5), WAYPOINTS, bc_type="natural")
    for s in np.linspace(0.0, 1.0, 11):
        np.testing.assert_allclose(kin.position([s], "cart"), path(4 * s), atol=1e-14)
        np.testing.assert_allclose(
            kin.jacobian([s], "cart").ravel(), 4 * path(4 * s, 1), atol=1e-13
        )
        np.testing.assert_allclose(
            kin.hessian([s], "cart")[:, 0, 0], 16 * path(4 * s, 2), atol=1e-11
        )
    np.testing.assert_allclose(kin.rotation([0.3], "cart"), np.eye(3), atol=1e-15)


def test_the_cart_starts_where_the_chain_puts_it_and_passes_every_waypoint():
    site = (0.0, 0.02, 0.5)
    kin = vmc.Kinematics(rail(site=site))
    np.testing.assert_allclose(kin.position([0.0], "cart"), site, atol=1e-15)
    for k in range(5):  # the displacement from the start is the path's, waypoint by waypoint
        moved = kin.position([k / 4], "cart") - kin.position([0.0], "cart")
        np.testing.assert_allclose(moved, WAYPOINTS[k] - WAYPOINTS[0], atol=1e-14)


def test_the_path_is_smooth_where_the_pieces_meet():
    kin = vmc.Kinematics(rail())
    for knot in (0.25, 0.5, 0.75):
        left, right = knot - 1e-9, knot + 1e-9
        np.testing.assert_allclose(
            kin.jacobian([left], "cart"), kin.jacobian([right], "cart"), atol=1e-7
        )
        np.testing.assert_allclose(
            kin.hessian([left], "cart"), kin.hessian([right], "cart"), atol=1e-6
        )


def test_two_waypoints_make_a_straight_rail_and_three_a_bend():
    straight = vmc.Kinematics(rail([[0.0, 0.0, 0.0], [0.2, 0.0, 0.4]]))
    np.testing.assert_allclose(straight.position([0.25], "cart"), [0.05, 0.0, 0.1], atol=1e-15)
    np.testing.assert_allclose(straight.hessian([0.25], "cart"), 0.0, atol=1e-15)
    bent = vmc.Kinematics(rail([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0]]))
    assert abs(bent.hessian([0.5], "cart")).max() > 0.1


def test_a_joint_after_the_rail_moves_with_the_cart():
    chain = SerialChain(
        [("rail", WAYPOINTS), "revolute"],
        axes=[None, [0.0, 1.0, 0.0]],
        points=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        sites={"cart": (1, [0.0, 0.0, 0.0]), "bob": (2, [0.0, 0.0, -0.3])},
    )
    path = CubicSpline(np.arange(5), WAYPOINTS, bc_type="natural")
    s, angle = 0.62, 0.4
    _, p = evaluate_frame(chain, np.array([s, angle]), "bob")
    swing = 0.3 * np.array([-np.sin(angle), 0.0, -np.cos(angle)])
    np.testing.assert_allclose(p, path(4 * s) + swing, atol=1e-14)


def test_the_rail_passes_the_model_contract_alone_and_with_a_joint_after_it():
    check_model(rail(), samples=3, energy=False)
    pendulum = SerialChain(
        [("rail", WAYPOINTS), "revolute"],
        axes=[None, [0.0, 1.0, 0.0]],
        points=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        sites={"bob": (2, [0.0, 0.0, -0.3])},
    )
    check_model(pendulum, samples=3, energy=False)


def test_a_rail_survives_a_dict_round_trip_in_json_and_a_rail_alone_has_no_unit():
    chain = SerialChain(
        [("rail", WAYPOINTS), "prismatic"],
        axes=[None, [1.0, 0.0, 0.0]],
        points=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        sites={"tip": (2, [0.0, 0.0, 0.1])},
    )
    copy = from_dict(json.loads(json.dumps(chain.to_dict())))
    q = np.array([0.37, 0.2])
    for a, b in zip(evaluate_frame(copy, q, "tip"), evaluate_frame(chain, q, "tip"), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-15)
    assert copy.joints == ["rail", "prismatic"] and copy.q_unit == "rad"
    assert rail().q_unit == ""


def test_a_rail_needs_its_waypoints_as_rows_of_three():
    for bad in (None, [[0.0, 0.0, 0.0]], [0.0, 0.0, 0.0], [[0.0, 0.0], [1.0, 0.0]]):
        with pytest.raises(ValueError, match="rail"):
            SerialChain(
                [("rail", bad)], axes=[None], points=[[0, 0, 0]], sites={"c": (1, [0, 0, 0])}
            )
    for spec in ("rail", ("rail", None)):
        with pytest.raises(ValueError, match="give a rail as"):
            SerialChain([spec], axes=[None], points=[[0, 0, 0]], sites={"c": (1, [0, 0, 0])})


def test_the_waypoints_are_a_design_param_of_the_chain_in_meters():
    waypoints = rail().params["j1.waypoints"]
    assert waypoints.shape == (5, 3) and waypoints.unit == "m" and waypoints.scope == "design"
    np.testing.assert_array_equal(waypoints.value, WAYPOINTS)


def bead(waypoints, mass=0.5):
    """A bead on the rail, which starts at the first waypoint, with gravity pulling down."""
    robot = vmc.Mechanism("bead", model=rail(waypoints, site=waypoints[0]))
    robot.add_param(vmc.Param("gravity", [0.0, 0.0, -9.81], unit="m/s^2"))
    robot.add("mass", vmc.PointMass(robot.point("cart"), mass))
    robot.add("gravity", vmc.Gravity(robot))
    return robot


def test_a_bead_on_a_circular_wire_swings_as_a_pendulum():
    # the wire runs well past the swing: a spline is flat at its ends, which a circle is not
    radius, reach = 0.5, np.radians(100.0)
    angles = np.linspace(-reach, reach, 17)
    circle = np.column_stack([radius * np.sin(angles), 0 * angles, radius * (1 - np.cos(angles))])
    robot = bead(circle)
    kin = vmc.Kinematics(robot)
    start = 0.8  # rad from the bottom
    q0 = (start + reach) / (2 * reach)  # the spline parameter is about the angle, uniformly
    plant = vmc.sim.ModelPlant(robot, q0=[q0], max_step=5e-5)
    times = np.linspace(0.0, 2.0, 21)
    swing = []
    for t in times:
        plant.advance(t - plant.t)
        x, _, z = kin.position(plant.q, "cart")
        swing.append(np.arctan2(x, radius - z))
    exact = solve_ivp(
        lambda t, y: [y[1], -9.81 / radius * np.sin(y[0])],
        (0.0, 2.0),
        [start, 0.0],
        t_eval=times,
        rtol=1e-11,
        atol=1e-12,
    )
    assert np.ptp(exact.y[0]) > 1.0  # it swings from side to side of the bottom
    np.testing.assert_allclose(swing, exact.y[0], atol=2e-3)


def test_a_bead_on_an_uneven_rail_keeps_its_energy_whatever_the_speed_of_the_path():
    # a straight incline sampled at the squares of 0..4: the speed of the path in s changes
    # about tenfold along it, so the mass in s does as much
    x = 0.0375 * np.arange(5) ** 2
    waypoints = np.column_stack([x, 0 * x, -0.5 * x])
    robot = bead(waypoints)
    kin, mass = vmc.Kinematics(robot), 0.5

    def energy(q, v):
        x, xdot = kin.position(q, "cart"), kin.jacobian(q, "cart") @ v
        return 0.5 * mass * xdot @ xdot + mass * 9.81 * x[2]

    assert np.linalg.norm(kin.jacobian([0.9], "cart")) > 8 * np.linalg.norm(
        kin.jacobian([0.0], "cart")
    )
    plant = vmc.sim.ModelPlant(robot, q0=[0.05], max_step=1e-4)
    start, drift, fastest = energy(plant.q, plant.v), [], 0.0
    for _ in range(8):
        plant.advance(0.05)
        drift.append(abs(energy(plant.q, plant.v) - start))
        fastest = max(fastest, np.linalg.norm(kin.jacobian(plant.q, "cart") @ plant.v))
    assert fastest > 1.0 and 0.5 < plant.q[0] < 0.95  # it slid fast, and stayed on the rail
    assert max(drift) < 5e-3 * mass * 9.81 * 0.1  # a tenth of the drop it fell


def test_live_waypoints_scale_the_path_and_so_the_mass():
    robot = bead(WAYPOINTS)
    dyn = compile_dynamics(robot, runtime=["bead.j1.waypoints"])
    assert dyn.live == ["bead.j1.waypoints"]
    q = np.array([0.3])
    p = dyn.live_values()
    scaled = np.asarray(p).ravel() * 2.0
    M, M2 = float(dyn.mass(q, p)), float(dyn.mass(q, scaled))
    assert M2 == pytest.approx(4.0 * M, rel=1e-12)
