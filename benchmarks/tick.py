"""Per-tick cost of a compiled soft-arm controller: 3 springs + 3 dampers on the 145/290/290 arm.

Run from the repository root: ``python benchmarks/tick.py``. Target: ≤ 30 µs per tick.
"""

import time

import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx


def build() -> vmc.Compiled:
    """The benchmark controller, compiled."""
    arm = helyx.arm("145-290-290")
    ctrl = vmc.Mechanism("ctrl")
    for i, s in enumerate((0.2, 0.6, 1.0)):
        point = arm.point(s=s)
        goal = vmc.Ref("goal", 3, value=[0.02, 0.0, 0.7 * s])
        ctrl.add(f"k{i}", vmc.LinearSpring(point - goal, 50.0))
        ctrl.add(f"c{i}", vmc.LinearDamper(point, 2.0))
    return vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl))


def per_tick(step, inputs, n: int = 5000) -> float:
    """Mean wall time of ``step(input)`` [µs] after a warm-up."""
    for k in range(300):
        step(inputs[k % len(inputs)])
    t0 = time.perf_counter()
    for k in range(n):
        step(inputs[k % len(inputs)])
    return (time.perf_counter() - t0) / n * 1e6


def main() -> None:
    """Print compile time and per-tick costs."""
    t0 = time.perf_counter()
    law = build()
    build_ms = (time.perf_counter() - t0) * 1e3
    rng = np.random.default_rng(0)
    theta, theta_dot = rng.uniform(-0.3, 0.3, (200, 9)), rng.uniform(-1.0, 1.0, (200, 9))
    x = [
        np.concatenate([a, b, law.live_values(), [0.0]])
        for a, b in zip(theta, theta_dot, strict=True)
    ]
    meas = [
        vmc.Signals(0.0, motor_position=a, motor_velocity=b)
        for a, b in zip(theta, theta_dot, strict=True)
    ]
    controller = vmc.VMCController(law)
    print(f"compile          : {build_ms:6.1f} ms")
    print(f"fast function    : {per_tick(lambda v: np.asarray(law.fast(v)), x):6.1f} µs/tick")
    print(f"controller.step  : {per_tick(lambda m: controller.step(0.0, m), meas):6.1f} µs/tick")


if __name__ == "__main__":
    main()
