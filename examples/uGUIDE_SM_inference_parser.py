import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time
import matplotlib.pyplot as plt
from pathlib import Path

from uGUIDE.utils import create_config_uGUIDE, load_config_uGUIDE, save_config_uGUIDE
from uGUIDE.data_utils import preprocess_data
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
parser.add_argument('--bvals', type=str,
                    help='File name containing the b-values.')
args = parser.parse_args()

if args.inference:
    if args.x_train is None or args.theta_train is None:
        raise ValueError('No training dataset given for the inference.')

theta_test = pd.read_csv(args.theta_test, header=None).values
x_test = pd.read_csv(args.x_test, header=None).values
bvals = np.loadtxt(args.bvals)
theta_test, x_test = preprocess_data(theta_test, x_test, bvals)

if Path.exists(Path.cwd() / 'config.pkl'):
    config = load_config_uGUIDE('config.pkl')
else:
    prior = {'f': [0.0, 1.0],
            'Da': [0.1, 3.0],
            'ODI': [0.03, 0.95],
            'u0': [0.0, 1.0],
            'u1': [0.0, 1.0]}
    config = create_config_uGUIDE(microstructure_model_name='Standard_Model',
                                size_x=x_test.shape[1],
                                prior=prior,
                                prior_postprocessing=None,
                                nf_features=6,
                                nb_samples=50_000,
                                max_epochs=200,
                                random_seed=1234)
    save_config_uGUIDE(config, savefile='config.pkl')

if args.inference:
    theta_train = pd.read_csv(args.theta_train, header=None).values
    x_train = pd.read_csv(args.x_train, header=None).values

    theta_train, x_train = preprocess_data(theta_train, x_train, bvals)

    run_inference(theta_train, x_train, config=config,
                  plot_loss=False, load_state=False)

print('Estimation of the microstructure parameters from the test signals')
nb_theta = theta_test.shape[0]
start_time = time.time()
# No postprocessing, because we do a comparison with ground truth values that contain u0 and u1
estimates = Parallel(n_jobs=10)(delayed(estimate_microstructure)(x_test[i,:], config,
                                                                 plot=False)
                                                                 for i in np.arange(nb_theta))
stop_time = time.time()
print('Time to estimate parameters in all voxels:', stop_time - start_time)

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
col = np.where(mask_degeneracy == True, 'r', 'b')
for p, param in enumerate(config['prior'].keys()):
    plt.subplot(1,config['size_theta'],p+1)
    plt.plot(theta_test[:nb_theta,p], theta_test[:nb_theta,p],
             c='k', alpha=0.5)
    idx_valid = np.where(mask[:,p] == True)
    plt.scatter(theta_test[idx_valid[0],p], map[idx_valid[0],p],
                c=col[idx_valid[0],p])
    plt.xlabel(f'{param}')
    plt.xlim(config['prior'][param][0], config['prior'][param][1])
    plt.ylim(config['prior'][param][0], config['prior'][param][1])
plt.savefig(config['folder_path'] / 'plot_ground_truth_map.png')
