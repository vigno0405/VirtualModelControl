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
