"""Draw the README figure: a soft arm pulled by a spring, a finger pressing on a table."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol import viz
from virtualmodelcontrol.robots import adapt, helyx

viz.use_style(usetex=False, font_size=14)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5.2))

# a hanging soft arm whose tip is pulled to a goal
arm = helyx.add_dynamics(helyx.arm("145-290-290"))
goal = np.array([0.25, 0.0, 0.55])
ctrl = vmc.Mechanism("ctrl")
ctrl.add("reach", vmc.LinearSpring(arm.point(s=1.0) - goal, 600.0))
ctrl.add("damp", vmc.LinearDamper(arm.point(s=1.0), 5.0))
ctrl.add("gravity", vmc.GravityCompensation(arm))
plant = vmc.sim.ModelPlant(arm)
law = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(arm, ctrl)))
vmc.sim.run(plant, law, vmc.sim.SimClock(1 / 330), T=3.0)
viz.draw_robot(ax0, arm, np.zeros(9), color="0.8", label="at rest")
viz.draw_robot(ax0, arm, plant.q, label="controlled")
viz.draw_spring(ax0, vmc.Kinematics(arm).position(plant.q, 1.0), goal)
viz.draw_goal(ax0, goal, label="goal")
ax0.invert_yaxis()
viz.label_axes(ax0)
ax0.legend(loc="lower left", fontsize=12)
ax0.set_title("soft arm: a spring pulls the tip")

# a finger pressing on a table with 1 N: the goal sits 1 cm below the surface
finger = adapt.add_dynamics(adapt.finger())
gap = vmc.PlaneDistance(finger.point("tip"), normal=[0, 0, -1], origin=[0, 0, 0.06])
finger.add("table", vmc.ContactSpring(gap, 1e4))
finger.add("table_damping", vmc.ContactDamper(gap, 5.0))
press = np.array([0.0, 0.05, 0.07])
ctrl = vmc.Mechanism("ctrl")
ctrl.add("press", vmc.LinearSpring(finger.point("tip") - press, 100.0))
ctrl.add("damp", vmc.LinearDamper(finger.point("tip"), 1.0))
ctrl.add("limits", adapt.joint_limit_spring(finger))
ctrl.add("gravity", vmc.GravityCompensation(finger))
plant = vmc.sim.ModelPlant(finger, q0=[0.8, 0.8], max_step=1e-4)
law = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(finger, ctrl)))
vmc.sim.run(plant, law, vmc.sim.SimClock(1 / 500), T=1.5)
tip = vmc.Kinematics(finger).position(plant.q, "tip")
force = 1e4 * max(0.0, tip[2] - 0.06)
ax1.axhspan(0.06, 0.075, color="0.88", zorder=0)
ax1.axhline(0.06, color="0.45", lw=1.2)
viz.draw_robot(ax1, finger, [0.8, 0.8], plane="yz", color="0.8", label="start")
viz.draw_robot(ax1, finger, plant.q, plane="yz", label="pressing")
viz.draw_goal(ax1, press, plane="yz", markersize=10, label="goal below the surface")
viz.draw_force(ax1, tip, [0.0, 0.0, -force], plane="yz", scale=0.02)
ax1.text(0.015, 0.068, "table", color="0.4", ha="center", va="center")
ax1.invert_yaxis()
viz.label_axes(ax1, "yz")
ax1.legend(loc="upper right", fontsize=12)
ax1.set_title(f"finger: contact force {force:.2f} N")

out = Path(__file__).resolve().parents[1] / "_static" / "readme-hero.png"
fig.savefig(out, dpi=110)
print(out, f"contact force {force:.3f} N")
