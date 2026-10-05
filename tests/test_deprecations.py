"""What leaves in 0.4.0 warns when used: the direct Dynamixel path and the ROS package. Hardware
profiles stay and do not warn."""

import importlib
import subprocess
import sys
import warnings

import pytest

import virtualmodelcontrol as vmc

HARDWARE = {
    "Bus": "bus",
    "FakeBus": "bus",
    "SdkBus": "bus",
    "latency_timer": "check",
    "scan": "check",
    "DynamixelPlant": "dynamixel",
    "home": "homing",
    "present_ticks": "homing",
}
ROS = {
    "TOPICS": "joint_io",
    "JointIO": "joint_io",
    "RosPlant": "plant",
    "LiveParams": "params",
    "serve": "twin",
    "control": "node",
}


@pytest.mark.parametrize("name", HARDWARE)
def test_the_direct_dynamixel_names_warn_and_still_work(name):
    with pytest.warns(
        DeprecationWarning, match=rf"hardware\.{name} is deprecated and leaves in 0\.4\.0"
    ):
        used = getattr(vmc.hardware, name)
    assert used is getattr(
        importlib.import_module(f"virtualmodelcontrol.hardware.{HARDWARE[name]}"), name
    )


@pytest.mark.parametrize("name", ["TOPICS", "JointIO"])
def test_the_ros_names_warn_and_still_work(name):
    with pytest.warns(
        DeprecationWarning, match=rf"ros\.{name} is deprecated and leaves in 0\.4\.0"
    ):
        used = getattr(vmc.ros, name)
    assert used is getattr(importlib.import_module(f"virtualmodelcontrol.ros.{ROS[name]}"), name)


@pytest.mark.parametrize("name", ["RosPlant", "LiveParams", "serve", "control"])
def test_the_ros_classes_warn_before_they_need_ros(name):
    with pytest.warns(DeprecationWarning, match=rf"ros\.{name} is deprecated"):
        try:
            getattr(vmc.ros, name)
        except ImportError:  # no rclpy here: the warning came first
            pass


@pytest.mark.parametrize(
    "source",
    [
        "from virtualmodelcontrol.hardware import FakeBus\n",
        "import virtualmodelcontrol as vmc\nvmc.hardware.scan\n",
        "from virtualmodelcontrol.ros import TOPICS\n",
        "import virtualmodelcontrol as vmc\nvmc.ros.JointIO\n",
    ],
    ids=["from-hardware", "attribute-hardware", "from-ros", "attribute-ros"],
)
def test_the_warning_points_at_the_line_that_used_the_name(source):
    with pytest.warns(DeprecationWarning) as record:
        exec(compile(source, "<a script>", "exec"), {})
    assert {w.filename for w in record} == {"<a script>"}


def test_scripts_see_the_warning_without_any_flag():
    code = "import virtualmodelcontrol as vmc; vmc.hardware.DynamixelPlant"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert (
        "DeprecationWarning: virtualmodelcontrol.hardware.DynamixelPlant is deprecated"
        in out.stderr
    )


def test_hardware_profiles_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name in ("HardwareProfile", "Motor", "KT", "MODES"):
            assert getattr(vmc.hardware, name) is not None
        from virtualmodelcontrol.hardware import HardwareProfile  # noqa: F401
        from virtualmodelcontrol.hardware.profile import HardwareProfile as Same  # noqa: F401


def test_the_modules_themselves_import_quietly():
    modules = ", ".join(
        f"virtualmodelcontrol.hardware.{m}" for m in ("bus", "check", "dynamixel", "homing")
    )
    subprocess.run([sys.executable, "-W", "error", "-c", f"import {modules}"], check=True)


def test_unknown_names_are_still_unknown():
    for package in (vmc.hardware, vmc.ros):
        with pytest.raises(AttributeError, match="has no attribute 'nothing'"):
            _ = package.nothing
