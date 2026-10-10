# API

Scripts start with `import virtualmodelcontrol as vmc`: the classes and functions of the
tutorials, and the modules `vmc.sim`, `vmc.viz` and the others below, are available from there.
The robot templates are the one exception: import them as
`from virtualmodelcontrol.robots import helyx` (or `adapt`, `bimanual`, `turtle`). The modules, from
the bottom layer up:

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   ~virtualmodelcontrol.core
   ~virtualmodelcontrol.math
   ~virtualmodelcontrol.mechanisms
   ~virtualmodelcontrol.models
   ~virtualmodelcontrol.system
   ~virtualmodelcontrol.compiler
   ~virtualmodelcontrol.dynamics
   ~virtualmodelcontrol.control
   ~virtualmodelcontrol.sim
   ~virtualmodelcontrol.adaptation
   ~virtualmodelcontrol.estimation
   ~virtualmodelcontrol.identification
   ~virtualmodelcontrol.optimization
   ~virtualmodelcontrol.hardware
   ~virtualmodelcontrol.viz
   ~virtualmodelcontrol.robots
   ~virtualmodelcontrol.config
   ~virtualmodelcontrol.testing
```
