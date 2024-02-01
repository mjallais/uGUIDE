import numpy as np
import torch
import random
from pathlib import Path
import pickle


def create_config_uGUIDE(microstructure_model_name,
                         size_x,
                         prior,
                         prior_postprocessing=None,
                         folderpath=None,
                         x_normalizer_file='x_normalizer.p',
                         theta_normalizer_file='theta_normalizer.p',
                         embedder_state_dict_file='torch_embedder_SM.pt',
                         nf_state_dict_file='torch_nf_SM.pt',
                         device=None,
                         nf_features=32,
                         hidden_layers=[128,64],
                         use_MLP=True,
                         learning_rate=1e-3,
                         max_epochs=500,
                         n_epochs_no_change=10,
                         random_seed=None,
                         nb_samples=50_000
                         ):
    """
    Create a configuration file for the inference and microstructure parameters
    estimation.

    Parameters
    ----------
    microstructure_model_name : str
                                Name of the model, used for saving the results
    
    size_x : int
            Size of the input data. Used to check inference and estimations
            are done on the same data.

    prior : dict
            Contains names of the model parameters used for inference with the
            minimum and maximum bounds of the Uniform prior distributions.
    
    prior_postprocessing : dict, optional
            Contains names of the model parameters after postprocessing with
            the minimum and maximum bounds of the parameters. This is useful 
            when the model parameters have constraints, making their prior
            distributions non-uniform, such as the Standard Model.
            If same as prior, set as ``None``.

    folderpath : str or Path, optional
            Path for saving the results. If None, default location will be 
            `results/uGUIDE_{microstructure_model_name}`.

    x_normalizer_file : str, default='x_normalizer.p'
            Name of the file for saving the normalizer of the input signal.

    theta_normalizer_file : str, defualt='theta_normalizer.p'
            Name of the file for saving the normalizer of the microstructure 
            parameters theta.
                         
    embedder_state_dict_file: str, default='torch_embedder_SM.pt'
            Name of the file for saving the embedded neural network after
            training.

    nf_state_dict_file : str, default='torch_nf_SM.pt'
            Name of the file for saving the normalizing flow after
            training.

    device : {'cpu', 'cuda'}, optional
            Device for running the inference and estimations. If ``None``,
            check if 'cuda' is available. If not, use 'cpu'.
    
    use_MLP : bool, default=True
            By default, use the MLP for dimension reduction. Set to False to 
            avoid dimensionality reduction and directly use the input signal.
            In this case, nf_features is set to the the size of the input 
            signal.

    nf_features : int, default=32
            Number of features extracted by the MLP. 
    
    hidden_layers : list, defaults=[128,64]
            Number of hidden units per layer for the MLP, used for the features
            extraction.

    learning_rate : float, default=1e-3
            Learning rate for the Adam optimizer.

    max_epochs : int, default=500
            Maximum number of epochs for the inference.
    
    n_epochs_no_change : int, default=10
            Number of epochs to wait for improvement on the validation set
            before stopping training.

    random_seed : int, optional
            Determines random number generation. Pass an int for reproducible
            results.
    
    nb_samples : int, default=50_000
            Number of samples drawn from the posterior distribution.

    Returns
    -------
    config : dict 
            Configuration file used for inference and microstructure parameters
            estimation.
    """
    config = {}

    # Set model name
    config['microstructure_model_name'] = microstructure_model_name
    
    # Save locations and name files of the neural networks
    if folderpath is None:
        folderpath = Path.cwd().parent / 'results' / f'uGUIDE_{microstructure_model_name}'
    folderpath.mkdir(exist_ok=True, parents=True)
    config['folderpath'] = folderpath
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
    # If user hasn't chosen between cpu and gpu, check if a gpu (cuda) is
    # available.
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
        else:
            config['device'] = 'cuda'
    elif device == 'cpu':
        config['device'] = device
    else:
        raise ValueError('Device not supported. Choose between cpu and cuda.')

    config['use_MLP'] = use_MLP
    if config['use_MLP'] == True:
        config['nf_features'] = nf_features
    else:
        config['nf_features'] = config['size_x']
    config['learning_rate'] = learning_rate
    config['max_epochs'] = max_epochs
    config['n_epochs_no_change'] = n_epochs_no_change
    config['hidden_layers'] = hidden_layers
    if random_seed is None:
        random_seed = random.randint(0, 10_000)
    config['random_seed'] = random_seed
    config['nb_samples'] = nb_samples

    return config


def save_config_uGUIDE(config, savefile='config.pkl', folderpath=None):
    """
    Save a configuration dictionary used for the inference and microstructure
    parameters estimation in a ``pkl`` file.

    Parameters
    ----------
    config : dict
        Dictionary containing μGUIDE configuration.

    savefile : str, default='config.pkl'
        Name of the file for saving μGUIDE configuration.

    folderpath : str or Path, optional
        Folder used for saving μGUIDE configuration. If ``None``, use 
        folderpath defined in μGUIDE configuration.
        
    """
    if folderpath is None:
        folderpath = config['folderpath']
    with open(folderpath / savefile, 'wb') as f:
        pickle.dump(config, f)
        print(f'uGUIDE config successfully saved to {folderpath / savefile}')
    
    return


def load_config_uGUIDE(savefile):
    """
    Load μGUIDE configuration used for the inference and microstructure
    parameters estimation stored in a ``pkl`` file.

    Parameters
    ----------
    savefile : str 
        Name of the file containing μGUIDE configuration.
    
    Returns
    -------
    config : dict
        Dictionary containing μGUIDE configuration.
    """

    with open(savefile, 'rb') as f:
        config = pickle.load(f)
        print('uGUIDE config successfully loaded.')
    
    return config
