# maximin_LHS
import sys
sys.path.append('../')
from fig1_calibration import create_sim as cs
from skopt.sampler import Lhs
import pandas as pd
import numpy as np
from scipy.stats import qmc

bounds = cs.define_pars(which='bounds', use_safegraph=True)
bounds['tn'][0] = 5.0 # make odds ratio lower from range of [10,60]
keys = list(bounds.keys())
reps = 25

lhs = Lhs(lhs_type='classic', criterion='maximin')
x = qmc.scale(
    lhs.generate([(0.0,1.0) for i in range(len(keys))], 50,random_state=42),
    l_bounds=[bounds[k][0] for k in keys],
    u_bounds=[bounds[k][1] for k in keys]
)

if __name__ == "__main__":
    df = pd.DataFrame(np.repeat(x,reps,axis=0),columns=keys)
    df['rand_seed'] = np.arange(len(df))
    df.to_csv('2024-05-21-maximin-LHS-n-50-r-25-tn-5.csv',index=False)
