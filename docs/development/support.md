# Support policy

What you can rely on from version 1.0.0 on.

## Versions of Python and of the packages

The library runs on Python 3.10 to 3.14, with numpy 1.26 or any 2.x. Every push runs the tests
on Ubuntu 22.04 and 24.04 with Python 3.10, 3.12 and 3.14. We have also run them from a fresh
clone on 3.11 and 3.13, and with the oldest versions the library allows: numpy 1.26, SciPy
1.11, CasADi 3.6 and matplotlib 3.8. PyTorch, which only the models exported as PyTorch code
need, is tried with its newest release.

A Python version is dropped, in a minor release and with a line in the
[changelog](changelog.md), once it has reached the end of its life. The same goes for the
oldest numpy, SciPy, CasADi and matplotlib that the library asks for.

The tests also run on macOS and Windows, on request: they are not part of every push.

## Versions of the library

Versions follow [semantic versioning](https://semver.org). A minor or patch release does not
break the public API, and a name that is going away warns with a `DeprecationWarning` for at
least one minor release before a major release removes it.

The public API is what the [reference](../api/index.md) lists: the names in the `__all__` of a
module. Names that start with an underscore, and modules the reference does not list, are
internal and may change at any time.

A fix can change numbers. When it does, the changelog says so, with what changed and from
which version, as it did for the arc formula of the soft arm's kinematics. An experiment that
must run the same way always has to pin the version ([Installation](../installation.md)).

Run logs carry the library's version and a version of their layout. A log of an older layout
is read by newer versions, and a log of a newer layout is refused with a message that says to
update. The changelog lists a change to the layout of the configuration files.

## Safety

The library gives no guarantee of safety on a real robot. Nothing limits a torque unless you add
the output stage that does, and a controller that is passive in theory can still go unstable
with the delay and the sampling of your loop. Read "Before the first run" in [Real-time
runs](../tutorials/real-time.md) before the first run on a robot.

## Problems

Report a bug, or a page that does not work as written, on the
[issue tracker](https://github.com/vigno0405/VirtualModelControl/issues), with the output of
`pip show virtualmodelcontrol` and of `python -c "import casadi, numpy; print(casadi.__version__, numpy.__version__)"`.
