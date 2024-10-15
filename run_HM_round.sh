#!/bin/bash
#BSUB -n 1 
#BSUB -R "(!gpu)" 
#BSUB -R "rusage[mem=16GB]"
#BSUB -J "array[1-50]" thisjobname 
#BSUB -W 5:00 
cd hetGPy-calibration
python3 run_LHS_timeseries.py -pm 2024-07-04-new_pars_wave003.csv -r 20 -i $LSB_JOBINDEX
