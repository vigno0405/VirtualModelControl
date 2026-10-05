"""The soft arm of the tutorials as a configuration: a spring to a goal, and a gentler controller
to swap to."""

import matplotlib
import pytest

matplotlib.use("Agg")

import virtualmodelcontrol as vmc

SPEC = {
    "robot": {"template": "helyx.arm", "geometry": "145-145-145"},
    "coordinates": {"tip": {"point": {"s": 1.0}}},
    "controller": {
        "elements": {
            "reach": {
                "type": "linear_spring",
                "coordinate": {
                    "difference": ["tip", {"ref": {"name": "goal", "value": [0.0, 0.0, 0.435]}}]
                },
                "stiffness": 300.0,
            },
            "damp": {"type": "linear_damper", "coordinate": "tip", "damping": 5.0},
            "gravity": {"type": "gravity_compensation"},
        }
    },
    "swaps": {
        "gentle": {
            "elements": {
                "reach": {
                    "type": "tanh_spring",
                    "coordinate": {"difference": ["tip", [0.0, 0.0, 0.40]]},
                    "stiffness": 300.0,
                    "max_force": 0.5,
                },
                "damp": {"type": "linear_damper", "coordinate": "tip", "damping": 5.0},
                "gravity": {"type": "gravity_compensation"},
            }
        }
    },
    "experiment": {
        "plant": {"type": "simulation", "dynamics": "helyx.add_dynamics"},
        "rate": 330,
        "duration": 1.0,
    },
}
DT = 1 / 330


@pytest.fixture
def make_experiment():
    """A new experiment on each call: controllers and plant are not shared."""
    return lambda: vmc.config.load(SPEC)


@pytest.fixture
def experiment(make_experiment):
    return make_experiment()
