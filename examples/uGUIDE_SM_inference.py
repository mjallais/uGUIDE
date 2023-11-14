#%%
import numpy as  np
# import torch
import pandas as pd
from pathlib import Path
from astropy import units

from uGUIDE.inference import run_inference

device = 'cpu'

#%%

folder_gen_train = Path.home().parents[1] / 'cubric' / 'data' / 'sapmj3' / 'SBI_dMRI' / 'generated_data' / 'uniform_distributions'
theta = pd.read_csv(folder_gen_train / "simulations_1e5_acq_MGH_CDMD_td_19_matlab_SM_uniform_De__f_Da_ODI_u0_u1.csv", header=None)
theta = theta.values

x = pd.read_csv(folder_gen_train / f"simulations_1e5_acq_MGH_CDMD_td_19_matlab_SM_uniform_De__S.csv", header=None)
x = x.values


#%%
run_inference(theta, x, device)