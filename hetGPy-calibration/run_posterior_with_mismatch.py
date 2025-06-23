
import os
os.environ['MPLCONFIGDIR'] = './cache'
os.environ['NUMBA_CACHE_DIR'] =  './cache'
import sys
sys.path.append('../')
from fig1_calibration import create_sim as cs
import sciris as sc
import pandas as pd


def make_and_run_sim(par_input):
    pars = par_input.copy()
    pars['verbose'] = False
    sim = cs.create_sim(pars,use_safegraph=True)
    df = cs.run_sim(sim = sim,
                    interactive = False, 
                    do_plot = False,
                    use_safegraph = True,
                    return_timeseries = False # return mismatch
    )
    for k,v in pars.items():
        df[k] = v
    return df

if __name__ == "__main__":

    parlist = pd.read_csv('posterior-samples.csv').to_dict(orient='records')
    ncpus = 50
    res = sc.parallelize(make_and_run_sim,parlist,ncpus=ncpus)
    df_out = pd.concat(res)
    df_out.to_csv(f'data/sims_combined-posterior-mismatch.csv')
    print("Ran new sims.")