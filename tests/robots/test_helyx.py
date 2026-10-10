import numpy as np
import pytest

from virtualmodelcontrol.models import evaluate_frame
from virtualmodelcontrol.robots import helyx


@pytest.mark.parametrize("geometry", list(helyx.GEOMETRIES))
def test_arm_defaults(geometry):
    arm = helyx.arm(geometry)
    L0 = helyx.GEOMETRIES[geometry]["L0"]
    assert {"seg1.L0", "seg3.delta", "seg2.r", "gravity", "m1.mass", "m3.s"} <= set(arm.params)
    np.testing.assert_allclose(evaluate_frame(arm.model, np.zeros(9), "tip")[1], [0, 0, sum(L0)])
    np.testing.assert_allclose(arm.params["gravity"].value, helyx.GEOMETRIES[geometry]["gravity"])
    b = arm.model.breakpoints()
    for i, length in enumerate(L0):
        assert float(arm.params[f"m{i + 1}.mass"].value) == pytest.approx(0.040 * length / 0.145)
        assert float(arm.params[f"m{i + 1}.s"].value) == pytest.approx((b[i] + b[i + 1]) / 2)
    assert helyx.ENCODER_SIGN[geometry] in (-1.0, 1.0)


def test_gravity_override_and_unknown_geometry():
    arm = helyx.arm(gravity=[0.0, 9.81, 0.0])
    np.testing.assert_allclose(arm.params["gravity"].value, [0.0, 9.81, 0.0])
    with pytest.raises(KeyError, match=r"known: .*145-145-145.*145-290-290.*290-145-145"):
        helyx.arm("100-100-100")
    with pytest.raises(KeyError, match="unknown geometry '290-145-145'"):
        helyx.hardware("290-145-145")  # the motors of that arm are in bimanual
