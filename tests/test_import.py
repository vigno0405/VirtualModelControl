import subprocess
import sys

import virtualmodelcontrol as vmc


def test_version_is_set():
    assert isinstance(vmc.__version__, str) and vmc.__version__


def test_import_never_pulls_in_ros():
    code = "import sys, virtualmodelcontrol; assert 'rclpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_viz_loads_matplotlib_only_when_used():
    code = (
        "import sys, virtualmodelcontrol as vmc; assert 'matplotlib' not in sys.modules; "
        "vmc.viz.animate; assert 'matplotlib' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_hardware_loads_only_when_used():
    code = (
        "import sys, virtualmodelcontrol as vmc; "
        "assert 'virtualmodelcontrol.hardware' not in sys.modules; "
        "vmc.hardware.HardwareProfile; assert 'dynamixel_sdk' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_ros_is_reachable_without_loading_ros():
    code = (
        "import sys, virtualmodelcontrol as vmc; "
        "assert 'virtualmodelcontrol.ros' not in sys.modules; "
        "vmc.ros.TOPICS; assert 'rclpy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-W", "ignore::DeprecationWarning", "-c", code], check=True)


def test_scipy_loads_only_for_the_fits_that_use_it():
    code = (
        "import sys, virtualmodelcontrol as vmc; "
        "assert not any(m.split('.')[0] == 'scipy' for m in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
