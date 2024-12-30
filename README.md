# Code for "Improving Policy-Oriented Agent-Based Modeling with History Matching: A Case Study"

This repository includes the code for reproducing the results in the above-mentioned manuscript by O'Gara, Kerr, Klein, Binois, Garnett, and Hammond.


## Organization

The repository is based off of the one in the original work, see here:

https://github.com/amath-idm/controlling-covid19-ttq




It is organized as follows: 
- `hetGPy-calibration` contains the analysis code to calibrate the model as described in the manuscript.

It also relies on modules originally from Kerr et. al 2021, which are:

- `fig1_calibration` and `fig5_projections` are the main folders containing the code for reproducing each figure of the manuscript.
- `inputs` and `outputs` are folders containing the input data and the model-based outputs, respectively.

Note that these analyses create large data files, which cannot be uploaded to github. These data are archived via Zenodo.

## Installation and usage

Use `pip install -r requirements.txt` to install dependencies. A Docker image (used for the simulations in the paper) is available here:
https://hub.docker.com/r/dogara/covasim-py310


## Running the History Matching Rounds

- See the `run_HM_round.sh` file in this directory for a sample of how to run the history matching rounds. The two inputs to the python script `hetGPy-calibration/run_HM.py` are:
    - `r` the round number
    - `n` whether to run and save new simulations (defaults to True)
- We recommend running the simulations on a computing cluster.



