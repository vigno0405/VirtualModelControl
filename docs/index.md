# virtualmodelcontrol

`virtualmodelcontrol` is a Python library for Virtual Model Control (VMC).
A robot and its controller are both *mechanisms*: coordinates plus components that store,
dissipate or supply energy. The controller is a virtual mechanism attached to the robot; its
springs and dampers are placed where they act, not tuned as abstract gains.

One CasADi model serves the real-time controller, simulation and optimization.

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/install
getting-started/first-controller
getting-started/simulate
```

```{toctree}
:maxdepth: 2
:caption: How-to

how-to/add-a-model
how-to/extend
```

```{toctree}
:maxdepth: 2
:caption: Theory

theory/mechanisms
theory/pcc
```

```{toctree}
:maxdepth: 2
:caption: Robots

robots/helyx
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/conventions
reference/api
reference/changelog
```

```{toctree}
:maxdepth: 2
:caption: Development

development/architecture
development/contributing
```
