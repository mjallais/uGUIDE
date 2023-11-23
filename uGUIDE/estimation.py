import torch
import pyro.distributions as dist


from uGUIDE.normalization import load_normalizer
from uGUIDE.density_estimator import get_nf
from uGUIDE.embedded_net import get_embedded_net

def sample_posterior_distribution(x, config):
    # Only one observation at a time
    # Is it useful to do it on the gpu here?

    if x.ndim == 1:
        x = x.reshape(1,-1)

    if config['size_x'] != x.shape[1]:
        raise ValueError('x size set in config is different from the size ' \
                         'of x used for training')

    # Normalize data
    x_normalizer = load_normalizer(config['folder_path'] / config['x_normalizer_file'])
    x_norm = x_normalizer(x)
    x_norm = torch.from_numpy(x_norm).to(config['device'])

    nf = get_nf(input_dim=config['size_theta'],
                nf_features=config['nf_features'],
                folder_path=config['folder_path'],
                nf_state_dict_file=config['nf_state_dict_file'],
                load_state=True)
    embedded_net = get_embedded_net(input_dim=config['size_x'],
                                    output_dim=config['nf_features'],
                                    folder_path=config['folder_path'],
                                    embedder_state_dict_file=config['embedder_state_dict_file'],
                                    layer_1_dim=config['hidden_layers'][0],
                                    layer_2_dim=config['hidden_layers'][1],
                                    load_state=True)

    base_dist = dist.Normal(
        loc=torch.zeros((config['nb_samples'],) + (config['size_theta'],)),
        scale=torch.ones((config['nb_samples'],) + (config['size_theta'],))
    )
    transformed_dist = dist.ConditionalTransformedDistribution(base_dist, nf)

    embedding = embedded_net(x_norm.type(torch.float32))

    samples_norm = transformed_dist.condition(
            embedding
        ).sample()

    theta_normalizer = load_normalizer(config['folder_path'] / config['theta_normalizer_file'])
    samples = theta_normalizer.inverse(samples_norm)

    return samples


def estimate_theta():
    # Get MAP, degeneracy, uncertainty and ambiguity

    return

def estimate_max_a_posteriori():

    return
    

def estimate_ambiguity():

    return


def estimate_uncertainty():

    return


def is_degenerate():

    return