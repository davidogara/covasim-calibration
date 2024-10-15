# HM.py
# do history matching waves for covasim
# author: David O'Gara
# date: 2024-06-03

# Libraries
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from hetgpy.hetGP import hetGP
import sys
sys.path.append('./')
sys.path.append('../')
from fig1_calibration import create_sim as cs
import pandas as pd
import numpy as np
from scipy.stats import qmc
import pickle
from datetime import datetime
from predict import predict_grid, load_model
from sample_NI import sample_NI_maximin


class HM:
    '''
    History matching analysis object
    '''
    def __init__(self,datafile,
                 wave_num,cutoff,
                 tstride,
                 outputs,
                 prev_data_file,
                 prev_sims,
                 sampling_strategy=None,
                 covtype = "Matern5_2",
                 DEBUG = False):
        # init arguments
        self.i               = wave_num
        self.datafile        = datafile
        self.tstride         = tstride
        self.cutoff          = cutoff
        self.prev_data_file  = prev_data_file
        self.sampling_strategy = sampling_strategy
        self.outputs           = outputs
        self.prev_sims         = prev_sims
        self.covtype = covtype

        # globals
        self.bounds          = cs.define_pars(which='bounds', use_safegraph=True)
        self.bounds['tn'][0] = 5.0
        self.keys            = list(self.bounds.keys())
        self.DEBUG           = DEBUG
        self.today           = datetime.today().strftime("%Y-%m-%d")
        self.use_nugs        = True
        self.use_MD          = False
        
        return
        
    def rescale(self,data,bounds):
        '''
        Scale data from [0,1]^d hypercube to bounds
        
        Parameters
        ----------

        data: DataFrame of designs, in [0,1]^d
        bounds: dict of param:(lower, upper) for rescaling
        
        Returns
        -------
        d: data scaled by bounds

        '''
        lower = [bounds[k][0] for k in self.keys]
        upper = [bounds[k][1] for k in self.keys]
        datavals = qmc.scale(data[self.keys].values,
        l_bounds = lower,
        u_bounds = upper
        )
        d = data.copy()
        d[self.keys] = datavals
        return d

    def scale_zero_one(self,data,bounds):
        '''
        Scale data from bounds to [0,1]^d
        
        Parameters
        ----------

        data: DataFrame of designs
        bounds: dict of param:(lower, upper) for rescaling
        
        Returns
        -------
        d: data scaled to unit hypercube
        '''
        lower    = [bounds[k][0] for k in self.keys]
        upper    = [bounds[k][1] for k in self.keys]
        datavals = qmc.scale(
                    data[self.keys].values,
                    l_bounds = lower,
                    u_bounds = upper,
                    reverse=True)
        
        d            = data.copy()
        d[self.keys] = datavals
        return d

    def save_model(self,model,fn):
        '''
        Save hetGPy model for later use

        Parameters
        ----------

        model: hetGPy model

        fn: filename

        Returns
        -------
        None
        '''
        # add date to fn
        fn  = 'models/' + fn
        with open(fn,'wb') as content:
            pickle.dump(model,content)
    
    def univariate_implausibility(self,df,Yf):
        '''
        Calculate implausibility measure of a univariate output

        df: DataFrame with mean (m) and variance (sd2 + nugs) predictions
        Yf: observed (field) data

        Returns
        -------
        Implausibility measure
        
        '''
        num   = np.abs(df['m'].values - np.repeat(Yf,len(df)))
        denom = df['sd2'].values
        if self.use_nugs:
            denom += df['nugs'].values
        SD_OBS = 0.0
        MD     = 0.0
        if self.use_MD:
            MD = 0.1 * denom
        denom = (denom + SD_OBS + MD)**0.5

        return num/denom

    def make_new_pars(self,df_NI,bounds,n,rand_seed_offset,seed,i):
        '''
        Make design locations for the next round of covasim simulations

        Parameters
        ----------
        df_NI: DataFrame of current NROY space
        bounds: dict of parameter: (lower, upper) values
        n: number of new designs (50)
        rand_seed_offset: increment of random seeds
        i: current wave number

        Returns
        -------
        new_pars: DataFrame of new design locations (with replications)
        
        '''
        df_NI_rescaled = self.rescale(df_NI,bounds=bounds)
        df_NI_rescaled.to_csv(f'hm_waves/{self.today}-NI_pars_wave{i:03d}.csv',index=False)
        
        # maximin design of NI space
        sample = sample_NI_maximin(df_NI_rescaled[self.keys],n=n,seed=self.i)


        new_pars_NI = pd.DataFrame(sample,columns=self.keys)
        new_pars_NI = pd.DataFrame(np.repeat(new_pars_NI.values,20,axis=0),columns=self.keys)
        
        new_pars = new_pars_NI.copy()
        new_pars['rand_seed'] = range(0,len(new_pars))
        new_pars['rand_seed'] += rand_seed_offset
        new_pars.to_csv(f'hm_waves/{self.today}-new_pars_wave{i:03d}.csv',index=False)
        return new_pars


    def train_hetGP(self,X,Z):
        '''
        Wrapper function to train a hetGPy model

        Parameters
        ---------
        X: ndarray of design locations
        Z: ndarray of outputs

        Returns
        -------
        model: hetGPy model
        '''
        model = hetGP()
        model.mleHetGP(
            X  = X,
            Z = Z,
            covtype = self.covtype,
        lower = [0.1 for i in range(X.shape[1])],
        upper = [10 for i in range(X.shape[1])],
        maxit = 1000,
        settings = {'checkHom':True}
        )
        return model

    def train_emulator(self,df,tkeep,bounds):
        '''
        Train hetGP model on covasim outputs.
        This method will handle both time-series and univariate outputs.

        Parameters
        ----------
        df: DataFrame of design locations and outputs (unscaled)
        tkeep: vector of times to use for time-series outputs
        bounds: dict of param: (lower,upper) for model parameters

        Returns
        -------
        models: dict of hetGPy models, of the form output_name: model
        '''
        # make X and Z

        # parse time vector
        min_rand_seed  = df['rand_seed'].max()
        tvec           = np.arange(df.query('rand_seed==@min_rand_seed').shape[0])
        df['t']        = np.concatenate(
                            [tvec for _ in range(df.groupby(['wave','rand_seed']).count().shape[0])]
                        )
        
        df   = df.loc[df['t'].isin(tkeep)]
        cols = [k for k in self.keys]
        cols.append('t')
        if type(self.tstride)==list:
            X = self.scale_zero_one(data=df.loc[df['t']==df['t'].min()],bounds=bounds)[self.keys].values
            t = np.empty(shape=(X.shape[0],0))
        else:
            X = self.scale_zero_one(data=df,bounds=bounds)[self.keys].values
            t = np.concatenate(
                [np.linspace(0,1,len(tkeep)) 
                for _ in range(df['rand_seed'].nunique())
                ]).reshape(-1,1)
            
        X = np.hstack([X,t])
        # map output keys
        if type(self.tstride)==list:
            models = {}
            for t_point, output in zip(self.tstride,self.outputs):
                models[f"{output}_{t_point}"] = self.train_hetGP(X,df.query("t==@t_point")[output].values)
        else:
            models = {}
            for outkey in self.outputs:
                models[outkey] = self.train_hetGP(X,df[outkey].values)
        
        return models
    
    
    
    def get_last_round_NI_path(self,i):
        '''
        Get name of NROY dataset from previous round of history matching

        Parameters
        ----------
        i: wave number

        Returns
        -------
        filename d
        
        '''
        if self.prev_data_file is None:
            last_round_NI_path = f"hm_waves/{self.today}-NI_pars_wave{i-1:03d}.csv"
        else:
            last_round_NI_path = self.prev_data_file

        return last_round_NI_path
    
    def eval_univariate_emulator(self,models,tvec,bounds,rand_seed_offset,i,data):
        '''
        Evaluate emulator models on NROY space

        Parameters
        ----------
        models: dict of key(output name): models
        tvec: vector of times
        bounds: dict of paramater: (lower,upper) values
        rand_seed_offset: random seed offset (passed to make_new_ars)
        i: wave number
        data: Observed (field) data

        Returns
        -------
        None
        '''
        # remove implausible from time series
        df_NI_dict = {}
        kw = {'ppd':20,'n_cores':6,'batch_size':1_000,'d':4}
        if self.DEBUG:
            kw = {'ppd':5,'n_cores':1,'batch_size':10,'d':4}
        if i == 0 and self.sampling_strategy=='grid_NI':
            kw = {'ppd':40,'n_cores':6,'batch_size':1_000,'d':4}
        if i > 0 and self.sampling_strategy=='grid_NI':
            kw['use_custom_grid'] = True
            grid = qmc.scale(
                pd.read_csv(self.prev_data_file)[self.keys].values,
                l_bounds = [self.bounds[k][0] for k in self.keys],
                u_bounds = [self.bounds[k][1] for k in self.keys],
                reverse=True
            )
            print(f"NI grid is length: {grid.shape[0]}")
            kw['custom_grid'] =  grid 
        NI_preds = {}
        for key, model in models.items():
            print(f" > Predicting: {key}")
            df_preds          = predict_grid(model,tvec,**kw).sort_values(self.keys)
            
            NI_preds[f"{key}_NI"] = self.univariate_implausibility(df_preds,Yf=data[key])
            
        
        df_NI_full = df_preds[self.keys].copy()
        for key in NI_preds.keys():
            df_NI_full[key] = NI_preds[key]
        df_NI_full.to_csv(f'hm_waves/{self.today}-predictions-with-implausibility-wave{self.i:03d}.csv')
        
        df_NI  = df_NI_full.copy()
        for key in NI_preds.keys():
            df_NI = df_NI.loc[df_NI[key]<self.cutoff]

        # check overlap
        print(f" > Checking join:")
        
        print(f" > NI samples: {df_NI.shape[0]}")
        self.make_new_pars(df_NI,
                            bounds=bounds,
                            n = 50,
                            rand_seed_offset=rand_seed_offset,
                            seed = i,
                            i=i)
        print(" > Wave complete.")
        return


    def get_empirical_data(self,sims,tstride=None):
        ''''
        Preprocess observed data

        Parameters
        ----------
        sims: DataFrame of covasim simulations
        tstride: times, either a list of times to evaluate covasim at, or an int of evaluating the model every tstride ticks

        Returns
        -------
        sims: DataFrame of simulations
        tkeep: vector of times to evaluate model at
        tvec: vector of times to evaluate model at (scaled to [0,1])
        '''
        
        tmax      = (sims['rand_seed']==sims['rand_seed'].min()).sum() # number of days
        sims['t'] = np.concatenate(
            [np.arange(0,tmax) 
             for i in range(sims.groupby(['wave','rand_seed']).count().shape[0])]
            )
        if type(tstride)==list:
            # build up list of outputs at different timesteps
            sims_data = {}
            for timestep, output in zip(self.tstride,self.outputs):
                output = f"{output}_data"
                key = f"{output}_{timestep}"
                sims_data[key] = sims.query('t==@timestep')[output].values
            
            sims_data['rand_seed'] = [j for (i,j) in sims.groupby(['wave','rand_seed']).count().index]
            sims_data = pd.DataFrame(sims_data)
            print(sims_data)
            tkeep     = self.tstride
            tvec      = np.empty(shape=(0,1))
        
        if self.DEBUG:
            tstride  = 21
        if tstride=='end':
            sims_data = sims.groupby('rand_seed').tail(1)
            tkeep     = [sims['t'].max()]
            tvec      = np.empty(shape=(0,1))
        elif type(self.tstride)!=list:
            tkeep     = np.arange(30,tmax,tstride)
            tvec      = (tkeep -tkeep.min()) / (tkeep.max()-tkeep.min())
            tvec      = tvec.reshape(-1,1)
            sims_data = sims.loc[sims['t'].isin(tkeep)]
        return sims_data, tkeep, tvec

    def run(self):
        '''
        Run history matching analysis
        '''
        bounds    = self.bounds
        if self.prev_sims is not None:
            files = self.prev_sims + [self.datafile]
            sims = pd.concat(
                [pd.read_csv(file).assign(wave=i) for i,file in enumerate(files)]
            )
        else:
            sims      = pd.read_csv(self.datafile).assign(wave=0)
        
        sims_data, tkeep, tvec = self.get_empirical_data(sims,tstride=self.tstride)
        sims_data = sims_data.loc[sims_data['rand_seed']==sims_data['rand_seed'].max()]
        save = True
        data = {}
        if type(self.tstride)==int:
            for outkey in self.outputs:
                data[outkey] = sims_data[f'{outkey}_data'].values
            print(sims_data)
        elif type(self.tstride)==list:
            for t, outkey in zip(self.tstride,self.outputs):
                data[f"{outkey}_{t}"] = sims_data[f'{outkey}_data_{t}'].values
        
        # Run Wave
        self.i = self.i
        print(" > Prepped data, running emulator")
        if os.path.exists(f'models/hetGPy-iter_{self.i:03d}.pkl'):
            print("Model path exists, skipping training")
            model = load_model(f'models/hetGPy-iter_{self.i:03d}.pkl')
        else:
            model = self.train_emulator(sims, tkeep, bounds = bounds)
        if save:
            self.save_model(model,f'hetGPy-iter_{self.i:03d}.pkl')
        print(" > Trained emulator")
        if len(tkeep)==1 or type(self.tstride)==list:
            self.eval_univariate_emulator(model,
                                tvec,
                                bounds = bounds,
                                data=data,
                                rand_seed_offset=sims['rand_seed'].max(),
                                i = self.i)
        else:
            self.eval_emulator(model,
                                tvec,
                                bounds = bounds,
                                data=data,
                                rand_seed_offset=sims['rand_seed'].max(),
                                i = self.i)
        print(" > New sims ready to run")
        
        print(" > Done")