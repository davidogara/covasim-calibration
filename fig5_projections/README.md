# Utility Functions for Policy Plot

This folder contains several functions from the original covasim analysis, see here:
https://github.com/amath-idm/controlling-covid19-ttq/tree/main/fig5_projections


Specfically, it contains:

New files:
- `plot_policy.ipynb`: Creates the figure. Note that this relies on `fig5-abc.msims` which is a large file, and is archived via Zenodo.

Original files:

- `create_sim.py` -- configures the simulation for `run_fig5.py`.
- `run_fig5.py` -- runs the simulations for Fig. 5.
- `fig5.msims` -- cached simulation results (the output of `run_fig5.py`).
- `kc_data` -- additional data used for this analysis.