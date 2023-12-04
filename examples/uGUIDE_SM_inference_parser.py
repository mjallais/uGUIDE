import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt

from uGUIDE.utils import create_config_uGUIDE
from uGUIDE.inference import run_inference
from uGUIDE.estimation import estimate_microstructure

parser = argparse.ArgumentParser()
parser.add_argument('--inference', action='store_true',
                    help='Run inference.')
parser.add_argument('--x_train', type=str, default=None,
                    help='Training data.')
parser.add_argument('--theta_train', type=str, default=None,
                    help='Training data.')
parser.add_argument('--x_test', type=str,
                    help='Testing data.')
parser.add_argument('--theta_test', type=str,
                    help='Testing data.')
args = parser.parse_args()

folder_simu = Path.home().parents[1] / 'cubric' / 'data' / 'sapmj3' / 'SBI_dMRI' / 'generated_data' / 'uniform_distributions'

theta_test = pd.read_csv(folder_simu / args.theta_test, header=None).values
x_test = pd.read_csv(folder_simu / args.x_test, header=None).values

prior = {'f': [0.0, 1.0],
         'Da': [0.1, 3.0],
         'ODI': [0.03, 0.95],
         'u0': [0.0, 1.0],
         'u1': [0.0, 1.0]}
config = create_config_uGUIDE(microstructure_model_name='Standard_Model',
                              size_theta=theta_test.shape[1],
                              size_x=x_test.shape[1],
                              prior=prior,
                              nf_features=6,
                              nb_samples=50_000,
                              epochs=40,
                              device='cpu')

if args.inference:
    theta_train = pd.read_csv(folder_simu / args.theta_train, header=None).values
    x_train = pd.read_csv(folder_simu / args.theta_train, header=None).values

    # test: use only limited number of samples
    x_train = x_train[:10_000,:]
    theta_train = theta_train[:10_000,:]

    run_inference(theta_train, x_train, config=config,
                plot_loss=True, load_state=False)

nb_theta = 100
start_time = time.time()
estimates = Parallel(n_jobs=10)(delayed(estimate_microstructure)(x_test[i,:], config, plot=False)
                                                                 for i in np.arange(nb_theta))
stop_time = time.time()
print('Time to estimate all parameters in all voxels:', stop_time - start_time)

map = np.zeros((nb_theta,config['size_theta']))
mask = np.zeros((nb_theta,config['size_theta']), dtype=bool)
mask_degeneracy = np.zeros((nb_theta, config['size_theta']), dtype=bool)
uncertainty = np.zeros((nb_theta, config['size_theta']))
ambiguity = np.zeros((nb_theta, config['size_theta']))

for i in np.arange(nb_theta):
    map[i,:] = estimates[i][0]
    mask[i,:] = estimates[i][1]
    mask_degeneracy[i,:] = estimates[i][2]
    uncertainty[i,:] = estimates[i][3]
    ambiguity[i,:] = estimates[i][4]

plt.figure(figsize=(5*config['size_theta'],5))
for p, param in enumerate(config['prior'].keys()):
    plt.subplot(1,config['size_theta'],p+1)
    plt.plot(theta_test[:nb_theta,p], theta_test[:nb_theta,p], c='k', alpha=0.5)
    plt.scatter(theta_test[:nb_theta,p], map[:,p])
    plt.xlabel(f'{param}')
    plt.xlim(config['prior'][param][0], config['prior'][param][1])
    plt.ylim(config['prior'][param][0], config['prior'][param][1])
plt.savefig(config['folder_path'] / 'plot_ground_truth_map.png')
