# RLFEM

Reinforcement Learning Environment for FEM Simulations based on FreeCAD.

## Installation

Dependencies:

* `gym`
* `freecad`

As `freecad` is only available as a conda package, you should install RLFEM as:

```bash
conda create --name my_env freecad -c conda-forge
conda activate my_env
pip install gym
pip install git+https://github.com/hex41434/FreeCAD_RL_FEM_Simulator.git@master
```

## Documentation

The documentation is at <https://hex41434.github.io/FreeCAD_RL_FEM_Environment/>. To generate the documentation, you will need:

```bash
pip install mkdocs mkdocs-material mkdocstrings pymdown-extensions mknotebooks
```

To see the documentation locally:

```bash
mkdocs serve
```

To put to github:

```bash
mkdocs gh-deploy
```
