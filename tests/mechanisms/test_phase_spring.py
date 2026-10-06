"""The phase-modulated spring: a plain spring when its depth is zero, a force that is exactly minus
the gradient of its energy (so the flywheel feels the reaction), a saturating potential, steering,
and a flywheel loop that equals the lab's controller and stays passive."""

import casadi as ca
import numpy as np
import pytest

import virtualmodelcontrol as vmc
from helpers import component_function
from virtualmodelcontrol.robots import turtle

rng = np.random.default_rng(21)
K, DEPTH, PEAK, STEER, E_MAX = 2.0, 0.6, 0.7, 0.3, 0.5


def coordinate():
    """(e, φ) as the first two joints of a robot."""
    return vmc.Stack(vmc.Joint(0, "rad"), vmc.Joint(1, "rad"))


def spring(**kw):
    return vmc.PhaseSpring(coordinate(), kw.pop("stiffness", K), **kw)


def force_and_energy(component, e, phase):
    """(f_e, f_φ, V) of ``component`` at the deflection ``e`` and the phase ``phase``."""
    f, V = component_function(component, 2)([e, phase], [0.0, 0.0])[1::3]
    return float(f[0]), float(f[1]), float(V)


def energy(component, e, phase):
    return force_and_energy(component, e, phase)[2]


def samples(n=12):
    return zip(rng.uniform(-1.5, 1.5, n), rng.uniform(-4.0, 4.0, n), strict=True)


def test_with_no_depth_no_steer_and_no_limit_it_is_a_linear_spring():
    plain = vmc.LinearSpring(vmc.Joint(0, "rad"), K)
    reference = component_function(plain, 2)
    mine = spring(peak=PEAK)  # the peak matters only with a depth
    for e, phase in samples():
        f_e, f_phase, V = force_and_energy(mine, e, phase)
        want = reference([e, phase], [0.0, 0.0])
        assert f_e == pytest.approx(float(want[1][0]), rel=1e-12)
        assert V == pytest.approx(float(want[4]), rel=1e-12)
        assert f_phase == 0.0  # nothing pushes the phase


@pytest.mark.parametrize("limit", [None, E_MAX])
def test_the_force_is_minus_the_gradient_of_the_energy_on_both_entries(limit):
    mine = spring(depth=DEPTH, peak=PEAK, steer=STEER, side=-1.0, limit=limit)
    h = 1e-6
    for e, phase in samples():
        f_e, f_phase, _ = force_and_energy(mine, e, phase)
        d_e = (energy(mine, e + h, phase) - energy(mine, e - h, phase)) / (2 * h)
        d_phase = (energy(mine, e, phase + h) - energy(mine, e, phase - h)) / (2 * h)
        assert f_e == pytest.approx(-d_e, rel=1e-6, abs=1e-9)
        assert f_phase == pytest.approx(-d_phase, rel=1e-6, abs=1e-9)


def test_the_plain_stiffness_follows_the_phase_and_its_reaction_is_minus_half_k_prime_e_squared():
    mine = spring(depth=DEPTH, peak=PEAK, steer=STEER, side=1.0)
    k_bar = K * (1 + STEER)
    for e, phase in samples():
        stiffness = k_bar * (1 + DEPTH * np.cos(phase - PEAK))
        slope = -k_bar * DEPTH * np.sin(phase - PEAK)  # K'(φ)
        f_e, f_phase, V = force_and_energy(mine, e, phase)
        assert f_e == pytest.approx(-stiffness * e, rel=1e-12)
        assert f_phase == pytest.approx(-0.5 * slope * e**2, rel=1e-12)
        assert V == pytest.approx(0.5 * stiffness * e**2, rel=1e-12)
    # the stiffest at the peak, the softest half a turn later
    assert force_and_energy(mine, 1.0, PEAK)[0] == pytest.approx(-k_bar * (1 + DEPTH))
    assert force_and_energy(mine, 1.0, PEAK + np.pi)[0] == pytest.approx(-k_bar * (1 - DEPTH))


def test_the_saturating_form_is_the_plain_one_for_small_deflections():
    plain = spring(depth=DEPTH, peak=PEAK, steer=STEER)
    soft = spring(depth=DEPTH, peak=PEAK, steer=STEER, limit=E_MAX)
    for phase in (-2.0, 0.3, 1.9):
        e = 1e-3 * E_MAX
        for a, b in zip(
            force_and_energy(plain, e, phase), force_and_energy(soft, e, phase), strict=True
        ):
            assert b == pytest.approx(a, rel=1e-5)


def test_the_saturating_form_has_the_formulas_of_psi_and_psi_b():
    soft = spring(depth=DEPTH, peak=PEAK, steer=STEER, side=-1.0, limit=E_MAX)
    k_bar = K * (1 - STEER)
    for e, phase in samples():
        psi = E_MAX**2 * (np.sqrt(1 + e**2 / E_MAX**2) - 1)
        psi_b = 0.5 * e**2 * E_MAX**2 / (e**2 + E_MAX**2)
        want = k_bar * (psi + DEPTH * np.cos(phase - PEAK) * psi_b)
        assert energy(soft, e, phase) == pytest.approx(want, rel=1e-12)


def test_the_torque_saturates_at_k_times_one_plus_steer_times_the_limit_whatever_the_depth():
    for depth in (0.0, 0.5, 0.9):
        for side, steer in ((1.0, 0.4), (1.0, -0.4), (-1.0, 0.4)):
            soft = spring(depth=depth, peak=PEAK, steer=steer, side=side, limit=E_MAX)
            k_bar = K * (1 + side * steer)
            for phase in (0.0, 1.0, 2.5):
                f_e, f_phase, _ = force_and_energy(soft, 1e4 * E_MAX, phase)
                assert f_e == pytest.approx(-k_bar * E_MAX, rel=1e-6)
                assert force_and_energy(soft, -1e4 * E_MAX, phase)[0] == pytest.approx(
                    k_bar * E_MAX, rel=1e-6
                )
                # a jammed limb still pushes on the flywheel, with ψ_b = e_max²/2 at most
                bound = 0.5 * k_bar * depth * np.sin(phase - PEAK) * E_MAX**2
                assert f_phase == pytest.approx(bound, rel=1e-6, abs=1e-12)
            top = K * (1 + abs(steer))
            for e in np.linspace(-30.0, 30.0, 241) * E_MAX:  # never above the limit's torque
                assert abs(force_and_energy(soft, e, 0.0)[0]) <= top * E_MAX


def test_a_positive_steer_stiffens_one_side_and_softens_the_other():
    right = spring(steer=0.4, side=1.0)
    left = spring(steer=0.4, side=-1.0)
    neutral = spring(steer=0.0, side=-1.0)
    e = 0.1
    assert force_and_energy(right, e, 0.0)[0] == pytest.approx(-K * 1.4 * e)
    assert force_and_energy(left, e, 0.0)[0] == pytest.approx(-K * 0.6 * e)
    assert force_and_energy(neutral, e, 0.0)[0] == pytest.approx(-K * e)
    assert energy(right, e, 0.0) > energy(neutral, e, 0.0) > energy(left, e, 0.0)
    # the opposite steer swaps them
    swapped = spring(steer=-0.4, side=1.0)
    assert force_and_energy(swapped, e, 0.0)[0] == pytest.approx(-K * 0.6 * e)


def test_the_defaults_are_a_stiffness_that_peaks_at_zero_phase_on_the_first_side_and_no_limit():
    mine = spring(depth=0.5)
    assert force_and_energy(mine, 1.0, 0.0)[0] == pytest.approx(-K * 1.5)  # peak 0
    assert force_and_energy(mine, 1.0, np.pi)[0] == pytest.approx(-K * 0.5)
    assert mine.steer.value == 0.0 and mine.side.value == 1.0 and mine.limit is None
    assert mine.peak.value == 0.0 and spring().depth.value == 0.0


def test_every_number_is_a_param_with_a_unit_and_a_scope():
    full = spring(depth=0.2, peak=0.1, steer=0.1, limit=0.4)
    params = full.params()
    assert list(params) == ["stiffness", "depth", "peak", "steer", "side", "limit"]
    scopes = {name: p.scope for name, p in params.items()}
    assert scopes == dict.fromkeys(("stiffness", "depth", "peak", "steer"), "stage") | {
        "side": "design",
        "limit": "episode",
    }
    assert params["stiffness"].unit == "N*m/rad" and params["limit"].unit == "rad"
    assert params["peak"].unit == "rad" and params["depth"].bounds == (0.0, 0.9)
    assert params["steer"].bounds == (-1.0, 1.0)
    assert spring().limit is None and "limit" not in spring().params()
    for module in (vmc, vmc.mechanisms, vmc.mechanisms.components):
        assert module.PhaseSpring is vmc.PhaseSpring and "PhaseSpring" in module.__all__
    assert vmc.core.registry.get("component", "phase_spring") is vmc.PhaseSpring
    a, b = vmc.Param("k", 3.0, unit="N*m/rad", scope="stage"), full.stiffness
    assert vmc.PhaseSpring(coordinate(), a).stiffness is a and a is not b  # a Param is shared


# The flywheel loop of the crawling turtle: two cranks that follow a virtual flywheel.
GAINS = dict(K=1.0, C=1e-3, Jv=0.1, bv=0.1, speed=2.0, delta=np.pi)
DT = 1 / 450


def cranks(inertia=0.002, damping=0.02):
    """The two cranks, each with its inertia and viscous friction: what the motors drive."""
    robot = turtle.robot()
    for i in range(2):
        robot.add(f"inertia{i}", vmc.Inertance(robot.joint(i), inertia))
        robot.add(f"friction{i}", vmc.LinearDamper(robot.joint(i), damping))
    return robot


def flywheel(robot, K, C, Jv, bv, speed, delta, ramp=0.0, spring=vmc.PhaseSpring, **args):
    """The virtual flywheel of the paper with a ``PhaseSpring`` between it and each crank."""
    ctrl = vmc.Mechanism("ctrl")
    phi = ctrl.add_state("flywheel", unit="rad")
    ctrl.add("flywheel", vmc.Inertance(phi, Jv))
    offset = vmc.Ref("phase", 1, value=delta, unit="rad")
    for i, (name, own, side) in enumerate((("left", phi, 1.0), ("right", phi - offset, -1.0))):
        e = robot.joint(i) - own
        ctrl.add(f"spring_{name}", spring(vmc.Stack(e, own), K, side=side, **args))
        ctrl.add(f"damper_{name}", vmc.LinearDamper(e, C))
    ctrl.add("drive", vmc.SpeedRegulator(phi, bv, speed, ramp))
    return ctrl


def controller_of(robot, ctrl):
    return vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))


def run(robot, controller, T, record=()):
    plant = vmc.sim.ModelPlant(robot, q0=[0.0, -np.pi])
    return vmc.sim.run(
        plant, controller, vmc.sim.SimClock(DT), T=T, z0=turtle.initial_state, record=record
    )


def lab_controller(robot):
    ctrl = turtle.controller(
        robot,
        stiffness=GAINS["K"],
        damping=GAINS["C"],
        inertia=GAINS["Jv"],
        flywheel_damping=GAINS["bv"],
        speed=GAINS["speed"],
        phase=GAINS["delta"],
        ramp_time=0.5,
    )
    return controller_of(robot, ctrl)


def test_with_no_depth_the_flywheel_loop_gives_the_labs_motor_torques():
    robot = cranks(damping=0.3)  # loaded, so that the cranks pull on the flywheel
    mine = controller_of(robot, flywheel(robot, **GAINS, ramp=0.5))
    lab = lab_controller(robot)
    first = vmc.Signals(0.0, motor_position=[0.3, -2.8], motor_velocity=[0.0, 0.0])
    for controller in (mine, lab):
        controller.reset(0.0, first, z0=turtle.initial_state(first))
    for k in range(1, 60):  # the same random measurements, and the flywheels move together
        meas = vmc.Signals(
            k * DT, motor_position=rng.uniform(-3, 3, 2), motor_velocity=rng.uniform(-5, 5, 2)
        )
        np.testing.assert_allclose(
            mine.step(meas.t, meas)["motor_torque"],
            lab.step(meas.t, meas)["motor_torque"],
            rtol=1e-9,
            atol=1e-12,
        )
        np.testing.assert_allclose(mine.z, lab.z, rtol=1e-9, atol=1e-12)
    ours = run(robot, mine, 1.5).arrays()  # and the closed loop, step after step
    theirs = run(robot, lab, 1.5).arrays()
    assert np.abs(theirs["motor_torque"]).max() > 0.05  # the cranks do get driven
    np.testing.assert_allclose(ours["motor_torque"], theirs["motor_torque"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(ours["motor_position"], theirs["motor_position"], rtol=1e-7)


class NoReaction(vmc.PhaseSpring):
    """The same spring without the force on the phase: the mistake the reaction guards against."""

    def force(self, ctx, y, yd):
        f = super().force(ctx, y, yd)
        return ca.vertcat(f[0], 0.0)


def balance_of(spring_class, damping, T):
    robot = cranks(damping=damping)
    ctrl = flywheel(robot, **GAINS, depth=0.6, peak=0.4, steer=0.3, spring=spring_class)
    log = run(robot, controller_of(robot, ctrl), T, record=["energy"])
    return vmc.sim.energy_balance(log)


def test_with_a_depth_the_flywheel_loop_stays_passive():
    balance = balance_of(vmc.PhaseSpring, damping=0.02, T=25.0)
    assert balance["margin"].min() >= 0.0  # it never gives more than it holds and is supplied
    assert balance["margin"].max() > 0.05  # and the check does see the energy move
    assert np.abs(balance["injected"]).max() < 1e-3 * balance["supplied"][-1]  # the books close


def test_without_the_reaction_on_the_phase_the_books_do_not_close():
    """Under a heavy load the reaction matters; the spring without it injects energy."""
    ok = np.abs(balance_of(vmc.PhaseSpring, damping=0.5, T=8.0)["injected"]).max()
    broken = balance_of(NoReaction, damping=0.5, T=8.0)
    assert ok < 1e-3 * broken["supplied"][-1]
    assert np.abs(broken["injected"]).max() > 1e-2 * broken["supplied"][-1]


def test_a_depth_changes_the_closed_loop_against_the_plain_spring():
    robot = cranks(damping=0.3)
    plain = run(robot, controller_of(robot, flywheel(robot, **GAINS)), 2.0).arrays()
    waved = run(robot, controller_of(robot, flywheel(robot, **GAINS, depth=0.6)), 2.0).arrays()
    change = np.abs(plain["motor_torque"] - waved["motor_torque"]).max()
    assert change > 0.05 * np.abs(plain["motor_torque"]).max()


def test_the_phase_spring_is_built_from_a_file_and_survives_saving_and_loading(tmp_path):
    path = tmp_path / "crawl.yaml"
    path.write_text(FILE)
    experiment = vmc.config.load(path)
    assert isinstance(experiment.mechanism.components["spring_left"], vmc.PhaseSpring)

    robot = cranks()
    args = dict(depth=0.5, peak=0.4, steer=0.2, limit=0.6)
    mine = controller_of(robot, flywheel(robot, **GAINS, ramp=0.5, **args))
    start = vmc.Signals(0.0, motor_position=[0.2, -2.9], motor_velocity=[1.0, 2.0])

    def torques(controller):
        controller.reset(0.0, start, z0=[0.5, 0.0])
        steps = (
            vmc.Signals(t, motor_position=[t, -3.0 + t], motor_velocity=[1, 1.5])
            for t in (0.1, 0.4, 0.9)
        )
        return [controller.step(m.t, m)["motor_torque"].tolist() for m in steps]

    from_file = experiment.controllers["ctrl"]
    assert torques(from_file) == torques(mine)

    elements = experiment.mechanism.components
    elements["spring_left"].depth.value = 0.7
    elements["spring_left"].steer.value = -0.1
    elements["spring_right"].limit.value = 0.45
    again = vmc.config.load(experiment.save(tmp_path / "tuned.yaml"))
    saved = again.to_dict()["controller"]["elements"]
    assert saved["spring_left"]["depth"] == 0.7 and saved["spring_left"]["steer"] == -0.1
    assert saved["spring_right"]["limit"] == 0.45 and saved["spring_right"]["side"] == -1.0
    kept = again.mechanism.components
    assert kept["spring_left"].depth.value == 0.7 and kept["spring_right"].limit.value == 0.45
    live = again.controllers["ctrl"].live_params()
    assert live["ctrl.spring_left.depth"] == 0.7 and live["ctrl.spring_right.depth"] == 0.5
    assert torques(again.controllers["ctrl"]) != torques(mine)  # the saved values are the new ones


FILE = """
robot: turtle.robot
controller:
  states:
    flywheel: {dim: 1, unit: rad, initial: 0.0}
  elements:
    flywheel: {type: inertance, coordinate: flywheel, inertance: 0.1}
    spring_left:
      type: phase_spring
      coordinate:
        stack: [{difference: [{joint: 0}, {state: flywheel}]}, {state: flywheel}]
      stiffness: 1.0
      depth: 0.5
      peak: 0.4
      steer: 0.2
      limit: 0.6
    spring_right:
      type: phase_spring
      coordinate:
        stack:
          - difference:
              - {joint: 1}
              - {difference: [{state: flywheel}, [3.141592653589793]]}
          - {difference: [{state: flywheel}, [3.141592653589793]]}
      stiffness: 1.0
      depth: 0.5
      peak: 0.4
      steer: 0.2
      side: -1.0
      limit: 0.6
    damper_left:
      type: linear_damper
      coordinate: {difference: [{joint: 0}, {state: flywheel}]}
      damping: 0.001
    damper_right:
      type: linear_damper
      coordinate:
        difference:
          - {joint: 1}
          - {difference: [{state: flywheel}, [3.141592653589793]]}
      damping: 0.001
    drive: {type: speed_regulator, coordinate: flywheel, gain: 0.1, speed: 2.0, ramp_time: 0.5}
"""
