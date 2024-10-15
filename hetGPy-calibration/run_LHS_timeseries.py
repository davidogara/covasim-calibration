import argparse
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from fig1_calibration import create_sim as cs
def make_and_run_sim(parlist,i):
    pars = parlist[i].copy()
    pars['verbose'] = False
    sim = cs.create_sim(pars,use_safegraph=True)
    df = cs.run_sim(sim = sim,
                    interactive = False, 
                    do_plot = False,
                    use_safegraph = True,
                    return_timeseries = True
    )
    df['t'] = np.arange(0,len(df))
    for k,v in pars.items():
        df[k] = v
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',type=int)
    parser.add_argument('-r',type=int)
    parser.add_argument('-pm')
    args = parser.parse_args()
    
    index        = int(args.i-1)
    pm           = args.pm
    runs_per_job = args.r
    rows_to_skip = int(index*runs_per_job)
    
    cols = pd.read_csv(pm,nrows=1).columns
    
    par_map = pd.read_csv(pm,
                          skiprows = rows_to_skip,
                          nrows=runs_per_job
                          )
    par_map.columns = cols
    parlist = par_map.to_dict(orient='records')
    res = [
        make_and_run_sim(parlist,i) for i in range(len(parlist))
    ]
    fp = f"temp/sims_{index:04d}.csv"
    pd.concat(res).to_csv(fp,index=False)
