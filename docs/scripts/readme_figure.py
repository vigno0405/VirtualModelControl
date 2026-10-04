"""Draw the README figure: the README's example, a soft arm pulled to a goal by a virtual spring.

Run from the repository root; writes docs/_static/readme-hero.svg (shown by the README) and .pdf.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import helyx

# The README's example: the same robot, controller and run.
arm = helyx.arm("145-290-290")
tip = arm.point(s=1.0)
goal = np.array([0.25, 0.0, 0.55])
ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(tip - goal, 600.0))
ctrl.add("damp", vmc.LinearDamper(tip, 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
controller = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
arm = helyx.add_dynamics(arm)
plant = vmc.sim.ModelPlant(arm)
rows = vmc.sim.run(plant, controller, vmc.sim.SimClock(dt=1 / 330), T=3.0).arrays()

kin = vmc.Kinematics(arm)
path = np.array([kin.position(q, 1.0) for q in rows["q"]])
distance = 100 * np.linalg.norm(path - goal, axis=1)  # [cm]

viz.use_style(usetex=False)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.8), width_ratios=(1, 1.25))
viz.draw_robot(ax0, arm, np.zeros(9), color="0.82", label="at rest")
ax0.plot(path[:, 0], path[:, 2], color=viz.PALETTE[0], lw=1.2, label="tip path")
viz.draw_robot(ax0, arm, rows["q"][-1], label="controlled")
viz.draw_spring(ax0, path[-1], goal)
viz.draw_goal(ax0, goal, label="goal")
ax0.invert_yaxis()  # the arm hangs: z points down
viz.label_axes(ax0)
ax0.legend(loc="upper right", fontsize=13)

ax1.plot(rows["t"], distance, color=viz.PALETTE[3])
ax1.set_xlabel("time [s]")
ax1.set_ylabel("tip to goal [cm]")
ax1.set_ylim(0, None)

out = Path(__file__).resolve().parents[1] / "_static" / "readme-hero"
print(*viz.save(fig, out), f"tip to goal at the end: {distance[-1]:.1f} cm")
