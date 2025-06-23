## Analysis files

This directory contains the notebooks to recreate the figures in the paper.

Note that these rely on a number of large data files, archived via Zenodo, which you will need to run these.

The figure notebooks are:
- Fig-01-Simulator-Emulator.ipynb (Figure 1)
- Fig-02-Optical-Depth.ipynb (Figure 2)
- Fig-03-S1-Wave-Designs-And-Time-Series (Figure 3, Figure S1)
- Fig-04-Posterior.ipynb (Figure 4)

And the analysis notebooks are:
- `calibrate-diagnoses.ipynb` and `calibrate-death.ipynb`: run ABC calibrations
- `joint-posterior.ipynb`: analyze ABC posterior
- `count-NROY-space.ipynb`: count size of history matching space (Table 1)

This directory also contains `posterior-samples.csv` which are the 50 parameterizations of the covasim posterior. 