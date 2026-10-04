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
