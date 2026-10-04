"""Every template is parametric in its geometry: arguments reach the model, defaults stay."""

import numpy as np
import pytest

import virtualmodelcontrol as vmc
from virtualmodelcontrol.robots import adapt, bimanual, helyx, ur5


def tip(robot, q, at):
    return vmc.Kinematics(robot).position(q, at)


def test_soft_arm_with_any_segments():
    arm = helyx.arm(
        lengths=(0.2, 0.3),
        section_radius=0.025,
        spool_radius=0.004,
        tendon_angles=np.radians([[0, 120, -120], [60, 180, -60]]),
        masses=(0.05, 0.06),
    )
    assert arm.model.space.nq == 6 and arm.actuation.motor_sizes(arm.model.space) == (6, 6)
    np.testing.assert_allclose(tip(arm, np.zeros(6), 1.0), [0, 0, 0.5], atol=1e-12)
    np.testing.assert_allclose(arm.model.breakpoints(), [0, 0.4, 1])
    assert float(arm.params["m2.mass"].value) == 0.06
    assert float(arm.params["seg1.d"].value) == 0.025 and float(arm.params["seg2.r"].value) == 0.004


def test_soft_arm_defaults_are_the_named_geometry():
    for name, spec in helyx.GEOMETRIES.items():
        named, explicit = helyx.arm(name), helyx.arm(name, lengths=spec["L0"])
        np.testing.assert_array_equal(named.params.vector(), explicit.params.vector())


def test_two_arms_with_their_own_lengths_and_bases():
    robot = bimanual.arms(
        lengths={"right": (0.27, 0.145, 0.145), "left": (0.29, 0.145, 0.145)},
        base_positions={"right": (0.15, 0.0, 0.0)},
        efficiency=0.2,
    )
    q = np.zeros(18)
    np.testing.assert_allclose(tip(robot, q, ("right", 1.0)), [0.15, 0, 0.56], atol=1e-12)
    np.testing.assert_allclose(tip(robot, q, ("left", 1.0)), [-0.125, 0, 0.58], atol=1e-12)
    assert float(robot.params["right.efficiency.c1"].value) == 0.2


def test_two_arms_dynamics_override_one_arm():
    robot = bimanual.add_dynamics(bimanual.arms(), stiffness={"left": np.full(9, 100.0)})
    np.testing.assert_array_equal(robot.params["left_stiffness.stiffness"].value, 100.0)
    np.testing.assert_array_equal(
        robot.params["right_stiffness.stiffness"].value, bimanual.STIFFNESS["right"]
    )


def test_finger_with_its_own_phalanges_and_transmission():
    finger = adapt.finger(
        link_lengths=(0.05, 0.03, 0.02), link_masses=(0.01, 0.01, 0.005), motor_radius=0.006
    )
    np.testing.assert_allclose(tip(finger, [0.0, 0.0], "tip"), [0, 0.1, 0], atol=1e-12)
    C = np.asarray(finger.model.coupling.value)
    np.testing.assert_allclose(C[0, 0], adapt.FINGER_PULLEY_RADIUS / 0.006)
    np.testing.assert_allclose(C[1, 1], 0.006 / adapt.PIP_TRANSMISSION)
    assert float(finger.params["m_dip.mass"].value) == 0.005
    np.testing.assert_array_equal(adapt.finger_coupling(), adapt.COUPLING)


def test_hand_overrides_one_entry_and_keeps_the_rest():
    default, custom = (
        adapt.hand(),
        adapt.hand(
            tip_offsets={"thumb": (-0.025, 0.0, 0.0)},
            finger_transmissions={"index": {"c_param": 0.012}},
            link_masses={"pinky_distal": 0.004},
        ),
    )
    q = np.zeros(13)
    assert not np.allclose(tip(custom, q, "thumb/tip"), tip(default, q, "thumb/tip"))
    for digit in ("index", "middle", "ring", "pinky"):
        np.testing.assert_array_equal(
            tip(custom, q, f"{digit}/tip"), tip(default, q, f"{digit}/tip")
        )
    C, C0 = np.asarray(custom.model.coupling.value), adapt.hand_coupling()
    np.testing.assert_allclose(C[6, 6], -0.005 / 0.012)  # index PIP
    np.testing.assert_array_equal(np.delete(C, [6, 7], axis=0), np.delete(C0, [6, 7], axis=0))
    assert float(custom.params["m_pinky_distal.mass"].value) == 0.004


def test_ur5_from_any_dh_table_and_the_hand_geometry_on_it():
    d, a = (0.1, 0, 0, 0.11, 0.095, 0.08), (0, -0.4, -0.4, 0, 0, 0)
    arm = ur5.arm(d=d, a=a)
    q = np.array([0.2, -0.9, 1.1, -0.4, 0.3, 0.1])
    reference = vmc.Mechanism("ref", model=vmc.models.SerialChain.from_dh(d, a, ur5.DH_ALPHA))
    np.testing.assert_allclose(tip(arm, q, "tool"), tip(reference, q, "tool"), atol=1e-14)
    both = ur5.with_hand(d=d, a=a, tip_offsets={"thumb": (0.0, 0.0, 0.02)})
    np.testing.assert_allclose(tip(both, np.r_[q, np.zeros(13)], "ur5/tool"), tip(arm, q, "tool"))


def test_segment_counts_must_agree_and_dynamics_follow_the_arm():
    with pytest.raises(ValueError, match="rows of tendon_angles"):
        helyx.arm(lengths=(0.2, 0.3))
    two = helyx.arm(lengths=(0.2, 0.3), tendon_angles=np.radians([[0, 120, -120]] * 2))
    with pytest.raises(ValueError, match="its own stiffness"):
        helyx.add_dynamics(two)
    helyx.add_dynamics(two, stiffness=np.full(6, 500.0), damping=np.full(6, 50.0))
    assert len(helyx.output_stage(6)[0].weights) == 6
    plant = vmc.sim.ModelPlant(two)
    plant.advance(0.01)
    assert np.all(np.isfinite(plant.q))


def test_two_arms_of_different_segment_counts_simulate():
    robot = bimanual.arms(
        lengths={"right": (0.2, 0.2)},
        tendon_angles={"right": np.radians([[0, 120, -120]] * 2)},
    )
    assert robot.model.space.nq == 15 and robot.actuation.motor_sizes(robot.model.space)[0] == 15
    robot = bimanual.add_dynamics(
        robot, stiffness={"right": [500.0] * 6}, damping={"right": [50.0] * 6}
    )
    plant = vmc.sim.ModelPlant(robot)
    plant.advance(0.01)
    assert np.all(np.isfinite(plant.q))
    with pytest.raises(ValueError, match="one row of tendon_angles"):
        bimanual.arms(lengths={"left": (0.2, 0.2)})


def test_joint_axes_and_hand_mounting_are_parameters():
    finger = adapt.finger(joint_axes=((0, 0, 1),) * 3)
    np.testing.assert_allclose(finger.params["j1.axis"].value, [0, 0, 1])
    q = np.zeros(19)
    near = vmc.Kinematics(ur5.with_hand()).position(q, "hand/index/tip")
    far = vmc.Kinematics(ur5.with_hand(mounting_position=(0, 0, 0.05))).position(
        q, "hand/index/tip"
    )
    tool = vmc.Kinematics(ur5.arm()).rotation(np.zeros(6), "tool")
    np.testing.assert_allclose(far - near, tool @ [0, 0, 0.05], atol=1e-12)
