#!/bin/bash
#BSUB -n 50 
#BSUB -R "(!gpu)" 
#BSUB -R "rusage[mem=128GB]"
#BSUB -W 5:00 
cd hetGPy-calibration
python3 run_HM.py
