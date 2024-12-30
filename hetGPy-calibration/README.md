## hetGPy calibration directory

This directory contains most of the files to recreate the analyses in the paper.

### Note on Data Availability
*   Due to the large file sizes, the `data`, `hm_waves`, and `models` directories are empty
*   These files are all archived via Zenodo at: 

The files and directories are described below:

*   `HistoryMatchingDictionary.csv`: contains the instructions for each round of history matching
*   `HM.py`: contains the history matching module
*   `run_HM.py`: the module to run the history matching analysis (works with `HM.py`) and run new simulations. Requires inputs for the history matching round `-r` and whether to run and save the simulations `-n` (default True)
*   `run_posterior.py`: run the ABC-calibrated posterior simulations
*  `sample_NI.py`: wrapper around the R `maximin` package to select a discrete maximin sample
*   `predict.py`: functions for gathering `hetGPy` predictions in parallel
*   `run_and_save_posterior_sims.py`: acts similarly to `run_posterior.py` but saves the actual covasim `sims` object (for easier plotting)

The `analysis` directory contains notebooks to create the figures in the paper.

Please note that we highly recommend running simulations on a computing cluster (we used 50 cores for the analyses in the paper).