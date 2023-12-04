#%%
%load_ext autoreload
%autoreload 2

#%%
import pandas as pd
from pathlib import Path

from uGUIDE.utils import create_config_uGUIDE
from uGUIDE.inference import run_inference
from uGUIDE.estimation import estimate_microstructure


#%%
folder_simu = Path.home().parents[1] / 'cubric' / 'data' / 'sapmj3' / 'SBI_dMRI' / 'generated_data' / 'uniform_distributions'
theta_train = pd.read_csv(folder_simu / 'simulations_1e6_acqMICRA_matlab_SM_uniform_De__f_Da_ODI_u0_u1.csv', header=None).values

snr = 50
x_train = pd.read_csv(folder_simu / f'simulations_1e6_acqMICRA_matlab_SM_uniform_De__S_snr_{int(snr)}.csv', header=None).values

#%%
# test: use only limited number of samples
x_train = x_train[:10_000,:]
theta_train = theta_train[:10_000,:]

#%%
prior = {'f': [0.0, 1.0],
         'Da': [0.1, 3.0],
         'ODI': [0.03, 0.95],
         'u0': [0.0, 1.0],
         'u1': [0.0, 1.0]}
config = create_config_uGUIDE(microstructure_model_name='Standard_Model',
                              size_theta=theta_train.shape[1],
                              size_x=x_train.shape[1],
                              prior=prior,
                              nf_features=6,
                              nb_samples=50_000,
                              epochs=40,
                              device='cpu')

#%%
run_inference(theta_train, x_train, config=config,
              plot_loss=True, load_state=False)


#%%
theta_test = pd.read_csv(folder_simu / 'simulations_1e4_acqMICRA_matlab_SM_uniform_De__f_Da_ODI_u0_u1.csv', header=None).values
x_test = pd.read_csv(folder_simu / f'simulations_1e4_acqMICRA_matlab_SM_uniform_De__S_snr_{int(snr)}.csv', header=None).values


#%%
_ = estimate_microstructure(x_test[0,:], config, plot=True)

# %%
