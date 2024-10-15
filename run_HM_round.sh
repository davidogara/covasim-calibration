#!/bin/bash
#BSUB -n 5 
#BSUB -R "(!gpu)" 
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 5:00 
cd hetGPy-calibration
python3 run_HM.py
