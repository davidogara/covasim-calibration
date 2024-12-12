import sys
sys.path.append('../')

import numpy as np
import sciris as sc
import covasim as cv
from fig1_calibration import create_sim as cs
import pandas as pd
T = sc.tic()

use_safegraph = 1
simsfile = 'posterior.sims' # Output file
ncpus = 50
parlist = pd.read_csv('posterior-samples.csv').to_dict(orient='records')
def make_sim(index):
    print(f'Now creating {index}:')
    entry = parlist[index]
    sc.pp(entry)
    pars = entry['pars']
    sim = cs.create_sim(pars=pars, use_safegraph=use_safegraph)
    sim.label = f'Trial {index}'
    sim.jsonpars = entry
    return sim


indices = [i for i in range(parlist)]
sims = sc.parallelize(make_sim, indices)

msim = cv.MultiSim(sims)
msim.run(par_args={'ncpus':ncpus})
for sim in msim.sims:
    sim.shrink()

cv.save(simsfile,msim)
sc.toc(T)
print('Done.')
