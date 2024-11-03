import os
os.environ['MPLCONFIGDIR'] = './cache'
os.environ['NUMBA_CACHE_DIR'] =  './cache'
from HM import HM
import pandas as pd
import numpy as np
import sciris as sc
import sys
sys.path.append('../')
from fig1_calibration import create_sim as cs
import argparse

def make_and_run_sim(par_input):
    pars = par_input.copy()
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

def parse_args():
    '''
    Set round number for analysis.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--round',type=int)

    args = parser.parse_args()
    return args.round

if __name__ == "__main__":
    run_new_sims = False
    if not os.getcwd().endswith('calibration'):
        os.chdir('hetGPy-calibration')
    filename = 'HistoryMatchingDictionary.csv'
    df       = pd.read_csv(filename)
    round    = parse_args()
    print(f"Running HM round {round}")
    pars = df.query("wave_num==@round").to_dict(orient='records')[0]
    pars['metric'] = "I_M"
    pars['tstride'] = [int(t) for t in pars['tstride'].split(",")]
    if len(pars['tstride'])==1:
        pars['tstride'] = int(pars['tstride'][0])
    pars['outputs'] = [name.strip() for name in pars['outputs'].split(",")]
    if pars['combine']:
        # stretch output by length of tstride
        ts = [pars['tstride']]*len(pars['outputs'])
        outs  = [pars['outputs']]*len(pars['tstride']) 
        pars['tstride'] = np.concatenate(ts).tolist() # flattens list
        pars['outputs'] = np.repeat(pars['outputs'],len(ts[0])).tolist()

    pars['prev_sims'] = pars['prev_sims'].split(',')
    pars['prev_sims'] = [] if pars['prev_sims'] == ['-'] else pars['prev_sims'] # come back here
    
    print(50*'*')
    print(f"Running:\n{pars}")
    history_match = HM(datafile       = pars['datafile'],
                       wave_num       = pars['wave_num'],
                       cutoff         = pars['cutoff'],
                       tstride        = pars['tstride'],
                       outputs        = pars['outputs'],
                       prev_data_file = pars['prev_data_file'],
                       prev_sims      = pars['prev_sims'],
                       metric         = pars['metric'],
                       sampling_strategy='grid_NI'
                       )
    history_match.run()

    if run_new_sims:
        new_designs = history_match.new_pars
        parlist = new_designs.to_dict(orient='records')
        ncpus = 50
        res = sc.parallelize(make_and_run_sim,parlist,ncpus=ncpus)
        df_out = pd.concat(res)
        df_out.to_csv(f'data/sims_combined-wave{round+1:03d}.csv')
        print("Ran new sims.")
    else:
        print("No new sims ran.")
