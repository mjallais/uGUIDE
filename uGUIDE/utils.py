import numpy as np
import torch
import random
from pathlib import Path
import pickle


def create_config_uGUIDE(microstructure_model_name, size_x, prior,
                         prior_postprocessing=None,
                         x_normalizer_file='x_normalizer.p',
                         theta_normalizer_file='theta_normalizer.p',
                         embedder_state_dict_file='torch_embedder_SM.pt',
                         nf_state_dict_file='torch_nf_SM.pt',
                         device=None, nf_features=32, learning_rate=1e-3,
                         max_epochs=500, random_seed=None,
                         n_epochs_no_change=10, hidden_layers=[128,64],
                         nb_samples=1_000):

    config = {}

    # Save locations and names
    folder_path = Path.cwd().parent / 'results' / f'uGUIDE_{microstructure_model_name}'
    folder_path.mkdir(exist_ok=True, parents=True)
    config['folder_path'] = folder_path
    config['x_normalizer_file'] = x_normalizer_file
    config['theta_normalizer_file'] = theta_normalizer_file
    config['embedder_state_dict_file'] = embedder_state_dict_file
    config['nf_state_dict_file'] = nf_state_dict_file

    # Microstructure model configuration
    config['size_theta'] = len(prior)
    config['size_x'] = size_x
    config['prior'] = prior
    if prior_postprocessing is None:
        config['prior_postprocessing'] = prior
    else:
        config['prior_postprocessing'] = prior_postprocessing

    # Inference configuration
    # If user hasn't chosen between cpu and gpu, check if a gpu (cuda) is available.
    # If so, use it. Otherwise, use the cpu.
    if device is None:
        if torch.cuda.is_available():
            config['device'] = 'cuda'
        else:
            config['device'] = 'cpu'
    elif device == 'cuda':
        if torch.cuda.is_available() == False:
            print('GPU usage requested, but cuda could not be found. Device ' \
                  'set to CPU instead')
            config['device'] = 'cpu'
    elif device == 'cpu' or device == 'cuda':
        config['device'] = device
    else:
        raise ValueError('Device not supported. Choose between cpu and cuda.')

    config['nf_features'] = nf_features
    config['learning_rate'] = learning_rate
    config['max_epochs'] = max_epochs
    config['n_epochs_no_change'] = n_epochs_no_change
    config['hidden_layers'] = hidden_layers
    if random_seed is None:
        random_seed = random.randint(0, 10_000)
    config['random_seed'] = random_seed

    # Sampling configuration
    config['nb_samples'] = nb_samples

    return config


def save_config_uGUIDE(config, savefile='config.pkl', folderpath=None):

    if folderpath is None:
        folderpath = config['folder_path']
    with open(folderpath / savefile, 'wb') as f:
        pickle.dump(config, f)
        print(f'uGUIDE config successfully saved to {folderpath / savefile}')
    
    return

def load_config_uGUIDE(savefile):

    with open(savefile, 'rb') as f:
        config = pickle.load(f)
        print('uGUIDE config successfully loaded.')
    
    return config
