#%%
%load_ext autoreload
%autoreload 2

#%%
import pandas as pd
from pathlib import Path

from uGUIDE.inference import run_inference
from uGUIDE.utils import create_config_uGUIDE
from uGUIDE.estimation import estimate_microstructure

#%%
theta_train = pd.read_csv('simulations_SM_train_10000__f_Da_ODI_u0_u1.csv', header=None).values
x_train = pd.read_csv(f'simulations_SM_train_10000__S_snr_50.csv', header=None).values

#%%
prior = {'f': [0.0, 1.0],
         'Da': [0.1, 3.0],
         'ODI': [0.03, 0.95],
         'u0': [0.0, 1.0],
         'u1': [0.0, 1.0]}
config = create_config_uGUIDE(microstructure_model_name='Standard_Model',
                              size_x=x_train.shape[1],
                              prior=prior,
                              nf_features=6,
                              nb_samples=50_000,
                              max_epochs=200,
                              random_seed=1234)

#%%
run_inference(theta_train, x_train, config=config,
              plot_loss=True, load_state=False)

#%%
theta_test = pd.read_csv('simulations_SM_test_1000__f_Da_ODI_u0_u1.csv', header=None).values
x_test = pd.read_csv(f'simulations_SM_test_1000__S_snr_50.csv', header=None).values

#%%
_ = estimate_microstructure(x_test[0,:], config, plot=True)

# %%
