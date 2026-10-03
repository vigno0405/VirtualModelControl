"""The hand template against values recorded from the original hand kinematics and controller."""

from pathlib import Path

import numpy as np

import virtualmodelcontrol as vmc
from virtualmodelcontrol.models import evaluate_frame
from virtualmodelcontrol.robots import adapt

DATA = np.load(Path(__file__).parents[1] / "data" / "hand.npz")
FINGER_LINKS = ("Spread", "MCP", "PIP", "DIP")
THUMB_LINKS = ("CMC1", "CMC2", "MCP", "IP")


def test_fingertips_match_the_recorded_hand():
    model = adapt.hand().model
    for i, q in enumerate(DATA["q"]):
        for digit in adapt.HAND_DIGITS:
            np.testing.assert_allclose(
                evaluate_frame(model, q, f"{digit}/tip")[1], DATA[f"{digit}_tip"][i], atol=1e-15
            )


def test_joint_frames_match_the_recorded_hand():
    model = adapt.hand().model
    chains = adapt.hand_coupling() @ DATA["q"].T  # joint angles (20, n)
    for digit, links, row in [("thumb", THUMB_LINKS, 0)] + [
        (f, FINGER_LINKS, 4 + 4 * k) for k, f in enumerate(("index", "middle", "ring", "pinky"))
    ]:
        chain = model.model.parts[digit]
        params = vmc.core.constants(chain.params)
        for i in range(len(DATA["q"])):
            joints = chains[row : row + 4, i]
            points = [np.array(p).ravel() for p in chain.joint_points(joints, params)]
            for k, link in enumerate(links):
                np.testing.assert_allclose(points[k], DATA[f"{digit}_{link}_p"][i], atol=1e-15)


def test_gravity_and_limit_torques_match_the_recorded_controller():
    robot, ctrl = adapt.hand(), vmc.Mechanism("ctrl")
    ctrl.add("gravity", vmc.GravityCompensation(robot))
    gravity = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl)))
    ctrl2 = vmc.Mechanism("ctrl")
    ctrl2.add("limits", adapt.hand_joint_limit_spring(robot))
    limits = vmc.VMCController(vmc.compile(vmc.VirtualMechanismSystem(robot, ctrl2)))
    for i, q in enumerate(DATA["q"]):
        meas = vmc.Signals(0.0, motor_position=q, motor_velocity=np.zeros(13))
        np.testing.assert_allclose(
            gravity.step(0.0, meas)["motor_torque"],
            DATA["gravity_torque"][i],
            rtol=1e-10,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            limits.step(0.0, meas)["motor_torque"], DATA["limit_torque"][i], rtol=1e-10, atol=1e-15
        )


def test_joint_frame_rotations_match_the_recorded_hand():
    model = adapt.hand().model
    # each link's mass site rides on one joint frame: compare that frame's rotation
    frames = {"thumb": {"2dof_joint": "CMC1", "proximal": "CMC2", "middle": "MCP", "distal": "IP"}}
    for f in ("index", "middle", "ring", "pinky"):
        frames[f] = {"proximal": "MCP", "middle": "PIP", "distal": "DIP"}
    for i, q in enumerate(DATA["q"]):
        for digit, links in frames.items():
            for link, joint in links.items():
                R = evaluate_frame(model, q, f"{digit}/{link}_cog")[0]
                np.testing.assert_allclose(R, DATA[f"{digit}_{joint}_R"][i], atol=1e-14)
