# large scale prediction with hetGPy
import numpy as np
from itertools import product
import pickle
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, parallel_config
import os
from time import time
keys  = ['beta','bc_lf','tn','bc_wc1']
parcols = keys.copy()
parcols.append('t')
def make_grid(ppd=5,d=4):
    '''
    Make a rectangular grid on [0,1]^d

    Parameters
    ----------
    ppd: points per dimension
    d: dimensions

    Returns
    -------
    grid: ndarray of grid locations
    '''
    xgrid = np.linspace(0,1,ppd) # points per dimension
    grid = np.array(
        list(
            product(*[xgrid for i in range(d)])
        )
    )
    return grid
def make_gridlist(grid,batch_size = 50):
    '''
    Transform a grid into a list of smaller grid for parallel prediction

    Parameters
    ----------
    grid: ndarray prediction grid
    batch_size: how many designs per entry in the output list
    
    Returns
    -------
    gridlist: grid separated into list entries
    '''
    batchlist = np.arange(0,grid.shape[0],batch_size)
    gridlist = []
    for i in range(len(batchlist)):
        if i < len(batchlist) - 1:
            s, e = batchlist[i], batchlist[i+1]
        else:
            s, e = batchlist[i], grid.shape[0]
        batch = grid[s:e,:]
        gridlist.append(batch)
    return gridlist

def load_model(fp):
    with open(fp,'rb') as stream:
        model = pickle.load(stream)
    return model

def predict_time_series(par,model):
    '''
    Call model.predict on set of parameters
    
    Parameters
    ----------
    par: ndarray prediction location for hetGPy
    model: hetGPy model

    Returns
    -------
    out: dictionary of hetGPy predictions
    '''
    preds = model.predict(par)
    out = {'m':preds['mean'],'sd2':preds['sd2'],'nugs':preds['nugs']}
    return out

def get_time_series_preds(model,tvec,xgrid):
    '''
    Compute grid predictions using hetGPy

    Parameters
    ----------
    model: hetGPy model
    tvec: vector of times (empty if predictions are univariate)
    xgrid: ndarray of grid locations

    Returns
    -------
    df: DataFrame of predictions at grid locations 
    
    '''
    if type(model)==str:
        model = load_model(model)
    s = time()
    if len(tvec)>1:
        parcols = keys + ['t']
        X = np.hstack([np.repeat(xgrid,len(tvec),axis=0),
        np.vstack([tvec]*xgrid.shape[0])])
    else:
        parcols = [key for key in keys]
        X = xgrid
    gridlist = make_gridlist(X,batch_size=1_000)
    l = []
    for Xg in gridlist:
        p = predict_time_series(Xg,model)
        df = pd.DataFrame(p)
        l.append(df)
    df = pd.concat(l)

    df[parcols] = X
    e = time()
    return df




def predict_grid(model,
                 tvec,
                 d = 4, 
                 ppd = 20,
                 batch_size = 50,
                 n_cores = 1,
                 inner_max_num_threads=2,
                 use_custom_grid = False,
                 custom_grid = None):
    '''
    Wrapper function for `get_time_series_preds` and allows for parallelization

    Parameters
    ----------
    model: hetGPy model
    tvec: vector of times (empty if outputs are univariate)
    d: dimension
    ppd: points per dimension
    batch_size: number of design points in each entry in gridlist
    n_cores: number of cores
    inner_max_num_threads: threads per parallel process
    use_custom_grid: boolean (use precomputed grid)
    custom_grid: grid used when use_custom_grid = True

    Returns
    -------
    results: DataFrame of predictive locations (with mean, sd2, and nugs)
    '''
    parallel = Parallel(n_jobs=n_cores)
    if use_custom_grid:
        grid = custom_grid
    else:
        grid     = make_grid(d=d,ppd=ppd)
    gridlist = make_gridlist(grid=grid,batch_size=batch_size)
    
    
    with parallel_config(backend="loky", inner_max_num_threads=inner_max_num_threads):
        results = parallel(delayed(get_time_series_preds)(model,tvec,gridlist[i])
        for i in tqdm(range(len(gridlist)))
        )

    return pd.concat(results)

if __name__ == "__main__":
    if 'hetGPy-calibration' not in os.getcwd():
        os.chdir('hetGPy-calibration')
    n_cores = 5

    parallel = Parallel(n_jobs=n_cores)
    
    #model = load_model('models/2024-05-07-hetGPy-iter_004.pkl')
    model = 'models/2024-05-07-hetGPy-iter_004.pkl' # try loading from fp internally
    tvec = np.linspace(0,1,125).reshape(-1,1)
    grid = make_grid(ppd=20)
    gridlist = make_gridlist(grid,batch_size=1_000)
    s = time()
    with parallel_config(backend="loky", inner_max_num_threads=2):
        results = parallel(delayed(get_time_series_preds)(model,tvec,gridlist[i])
        for i in tqdm(range(len(gridlist)))
        )
    e = time()
    print(round(e-s,2))
    out = results