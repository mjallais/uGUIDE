#%%
%load_ext autoreload
%autoreload 2

#%%
import numpy as  np
import pandas as pd
from pathlib import Path
from astropy import units

from uGUIDE.utils import create_config_uGUIDE
from uGUIDE.inference import run_inference
from uGUIDE.estimation import sample_posterior_distribution
from uGUIDE.plot_utils import plot_posterior_distribution


#%%
folder_gen_train = Path.home().parents[1] / 'cubric' / 'data' / 'sapmj3' / 'SBI_dMRI' / 'generated_data' / 'uniform_distributions'
theta = pd.read_csv(folder_gen_train / "simulations_1e5_acq_MGH_CDMD_td_19_matlab_SM_uniform_De__f_Da_ODI_u0_u1.csv", header=None)
theta = theta.values

x = pd.read_csv(folder_gen_train / f"simulations_1e5_acq_MGH_CDMD_td_19_matlab_SM_uniform_De__S.csv", header=None)
x = x.values

#%%
# test: use only limited number of samples
x_train = x[:10_000,:]
theta_train = theta[:10_000,:]

#%%
prior = {'f': [0.0, 1.0],
         'Da': [0.1, 3.0],
         'ODI': [0.03, 0.95],
         'u0': [0.0, 1.0],
         'u1': [0.0, 1.0]}
config = create_config_uGUIDE(microstructure_model_name='Standard_Model',
                              size_theta=theta.shape[1],
                              size_x=x.shape[1],
                              prior=prior,
                              nb_samples=100)

#%%
run_inference(theta_train, x_train, config=config,
              plot_loss=True, load_state=True)

# %%
samples = sample_posterior_distribution(x[-1,:], config)
plot_posterior_distribution(samples, config)

# %%
