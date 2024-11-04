from run_HM import make_and_run_sim
import sciris as sc
import pandas as pd
if __name__ == "__main__":

    parlist = pd.read_csv('posterior-samples.csv').to_dict(orient='records')
    ncpus = 50
    res = sc.parallelize(make_and_run_sim,parlist,ncpus=ncpus)
    df_out = pd.concat(res)
    df_out.to_csv(f'data/sims_combined-posterior.csv')
    print("Ran new sims.")