# Code for "Improving Policy-Oriented Agent-Based Modeling with History Matching: A Case Study"

This repository includes the code for reproducing the results in the manuscript 




## Organization

The repository is based off of the one in the original work, see here:

https://github.com/amath-idm/controlling-covid19-ttq


It is organized as follows: 
- `hetGPy-calibration` contains the analysis code to calibrate the model as described in the manuscript

It also relies on modules originally from Kerr et. al 2021, which are:

- `fig1_calibration` and `fig5_projections` are the main folders containing the code for reproducing each figure of the manuscript.
- `inputs` and `outputs` are folders containing the input data and the model-based outputs, respectively.

Note that these analyses create large data files, which cannot be uploaded to github. These data are archived via Zenodo.

## Installation and usage

Use `pip install -r requirements.txt` to install dependencies. A Docker image (used for the simulations in the paper) is available here:
https://hub.docker.com/r/dogara/covasim-py310




