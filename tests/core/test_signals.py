import numpy as np

from virtualmodelcontrol.core import Signals


def test_signals_store_validity():
    s = Signals(t=1.5, motor_position=[0.1, 0.2])
    s.set("motor_velocity", [np.nan, 0.0])
    s.set("force", 1.0, valid=False)
    assert s.t == 1.5 and "motor_position" in s and s.names[-1] == "force"
    assert s.is_valid("motor_position")
    assert not s.is_valid("motor_velocity")  # NaN
    assert not s.is_valid("force")  # flagged
    assert not s.is_valid("missing")
    assert s["force"].shape == (1,)
