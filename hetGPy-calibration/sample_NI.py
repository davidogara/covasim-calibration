# wrapper for maximin
import numpy as np
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.packages import importr
maximin = importr('maximin')
np_cv_rules = default_converter + numpy2ri.converter

def sample_NI_maximin(df,n=50,seed = None):
    # shuffle df
    idxs = np.random.default_rng(seed=seed).choice(df.shape[0],size=n)
    with np_cv_rules.context():
        d = maximin.maximin_cand(n=n,Xcand=df.values,Xorig = df.values[idxs,:])
        out_idxs = (d['inds']-1).astype(int)
    return df.values[out_idxs,:]

