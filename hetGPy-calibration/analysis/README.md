## Analysis files

This directory contains the notebooks to recreate the figures in the paper.

Note that these rely on a number of large data files, archived via Zenodo, which you will need to run these.

The notebooks are:
- `plot_wave_sims.ipynb`: Plot history matching rounds (Fig. 1) and wave design locations (Fig. 3)
- `optical-depth.ipynb`: Optical depth plots (Fig. 2)
- `calibrate-diagnoses.ipynb` and `calibrate-death.ipynb`: run ABC calibrations
- `joint-posterior.ipynb`: analyze ABC posterior
- `plot-posterior.ipynb`: Plot ABC posterior (Fig. 4)
- `count-NROY-space.ipynb`: count size of history matching space (Table 1)

This directory also contains `posterior-samples.csv` which are the 50 parameterizations of the covasim posterior. 