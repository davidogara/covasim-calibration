from HM import HM
import pandas as pd
import numpy as np
import os
if __name__ == "__main__":
    if not os.getcwd().endswith('calibration'):
        os.chdir('hetGPy-calibration')
    filename = 'HistoryMatchingDictionary-v2.csv'
    df       = pd.read_csv(filename)
    round    = 3

    pars = df.query("wave_num==@round").to_dict(orient='records')[0]
    
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
                       sampling_strategy='grid_NI'
                       )
    history_match.run()