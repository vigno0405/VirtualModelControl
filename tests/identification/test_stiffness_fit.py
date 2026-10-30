import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.identification import fit_stiffness_damping
from virtualmodelcontrol.robots import helyx


def test_stiffness_and_damping_are_recovered_from_a_simulated_run():
    K, D = 0.6 * helyx.SIM_STIFFNESS, 0.8 * helyx.SIM_DAMPING
    plant = vmc.sim.ModelPlant(helyx.add_dynamics(helyx.arm("145-145-145"), stiffness=K, damping=D))
    dt, zero, rng = 1 / 330, np.zeros(9), np.random.default_rng(1)
    for _ in range(330):  # settle under gravity: the run starts at rest
        plant.write(vmc.Signals(plant.t, motor_torque=zero))
        plant.advance(dt)
    rows, torque = [], zero
    for k in range(1200):
        if k >= 50 and k % 100 == 0:  # torque steps after a resting baseline
            torque = rng.uniform(-0.05, 0.15, 9)
        rows.append((plant.t, plant.q.copy(), plant.v.copy(), torque))
        plant.write(vmc.Signals(plant.t, motor_torque=torque))
        plant.advance(dt)
    t, q, v, u = (np.array(x) for x in zip(*rows, strict=True))
    known = helyx.arm("145-145-145")
    known.add("gravity", vmc.Gravity(known))
    K_fit, D_fit = fit_stiffness_damping(known, [dict(t=t, q=q, v=v, u=u)], smoothing=11)
    np.testing.assert_allclose(K_fit, K, rtol=0.05)
    np.testing.assert_allclose(D_fit, D, rtol=0.03)


def _two_masses(K, D):
    from virtualmodelcontrol.models import JointSpace

    robot = vmc.Mechanism("r", model=JointSpace(2, unit="m"))
    for i in range(2):
        robot.add(f"m{i}", vmc.Inertance(robot.joint(i), 1.0))
    both = robot.joint(slice(0, 2))
    robot.add("spring", vmc.LinearSpring(both, np.array(K)))
    robot.add("damper", vmc.LinearDamper(both, np.array(D)))
    return robot


def _run_of_the_first_mass():
    from virtualmodelcontrol.identification import Steps

    steps = Steps([0.0, 0.0], pulls=[(0, 2.0), (0, -2.0)], hold=2.0, rest=2.0)
    plant = vmc.sim.ModelPlant(_two_masses([40.0, 25.0], [1.5, 1.0]), max_step=1e-3)
    return vmc.sim.run(plant, steps, vmc.sim.SimClock(1 / 200), T=steps.duration)


def test_a_coordinate_that_never_moves_is_nan_with_a_warning_not_zero():
    import pytest

    log = _run_of_the_first_mass()
    guess = _two_masses([0.0, 0.0], [0.0, 0.0])
    with pytest.warns(UserWarning, match=r"stiffness 1, damping 1"):
        K, D = fit_stiffness_damping(guess, [log], smoothing=11)
    assert np.isnan(K[1]) and np.isnan(D[1])
    np.testing.assert_allclose([K[0], D[0]], [40.0, 1.5], rtol=0.05)


def test_the_smoothing_window_must_be_odd():
    import pytest

    log = _run_of_the_first_mass()
    guess = _two_masses([0.0, 0.0], [0.0, 0.0])
    for bad in (10, 12, 3):
        with pytest.raises(ValueError, match="odd number of samples"):
            fit_stiffness_damping(guess, [log], smoothing=bad)
