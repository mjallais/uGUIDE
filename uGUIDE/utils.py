from pathlib import Path

# Call file data_utils?

# Check if testing signal corresponds to observed signals used for training?

# Normalize signal wrt b_0 and remove nan and inf

def create_config_uGUIDE(microstructure_model_name, size_theta, size_x, prior,
                         x_normalizer_file='x_normalizer.p',
                         theta_normalizer_file='theta_normalizer.p',
                         embedder_state_dict_file='torch_embedder_SM.pt',
                         nf_state_dict_file='torch_nf_SM.pt',
                         device='cpu', nf_features=32, learning_rate=1e-3,
                         epochs=40, hidden_layers=[128,64],
                         nb_samples=1_000):

    config = {}

    # Save locations and names
    folder_path = Path.cwd() / f'uGUIDE_{microstructure_model_name}'
    folder_path.mkdir(exist_ok=True, parents=True)
    config['folder_path'] = folder_path
    config['x_normalizer_file'] = x_normalizer_file
    config['theta_normalizer_file'] = theta_normalizer_file
    config['embedder_state_dict_file'] = embedder_state_dict_file
    config['nf_state_dict_file'] = nf_state_dict_file

    # Microstructure model configuration
    config['size_theta'] = size_theta
    config['size_x'] = size_x
    config['prior'] = prior

    # Inference configuration
    config['device'] = device
    config['nf_features'] = nf_features
    config['learning_rate'] = learning_rate
    config['epochs'] = epochs
    config['hidden_layers'] = hidden_layers

    # Sampling configuration
    config['nb_samples'] = nb_samples

    return config


def save_config_uGUIDE(config):

    return