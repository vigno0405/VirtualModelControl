"""The soft arm held by one controller, and a saturating spring to plan in its place."""

import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import helyx

HORIZON, TRANSITION, NODES = 5.0, 2.0, 21
SCALES = (0.05, 0.3, 20.0)


def soft_arm_swap():
    """The hanging arm held by a linear spring, and a saturating spring to plan in its place."""
    arm = helyx.add_dynamics(helyx.arm("145-290-290"))
    tip = arm.point(s=1.0)
    hold = vmc.Mechanism("hold")
    hold.add("drag", vmc.LinearSpring(tip - [0.04, 0.0, 0.70], 10.0))
    hold.add("damper", vmc.LinearDamper(tip, 2.0))
    hold.add("gravity", vmc.GravityCompensation(arm))
    goal = vmc.Ref("goal", 3, [0.10, 0.04, 0.68])
    new = vmc.Mechanism("new")
    stiffness = vmc.Param("stiffness", 10.0, bounds=(1.0, 150.0), scope="stage")
    new.add("pull", vmc.TanhSpring(tip - goal, stiffness, 2.0))
    new.add("damper", vmc.LinearDamper(tip, 2.0))
    new.add("gravity", vmc.GravityCompensation(arm))
    return (
        arm,
        tip,
        goal,
        vmc.VirtualMechanismSystem(arm, hold),
        vmc.VirtualMechanismSystem(arm, new),
    )
