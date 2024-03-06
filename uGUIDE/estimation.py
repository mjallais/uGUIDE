import numpy as np
import torch
import pyro.distributions as dist
from scipy import optimize
import traceback
import matplotlib.pyplot as plt

from uGUIDE.normalization import load_normalizer
from uGUIDE.density_estimator import get_nf
from uGUIDE.embedded_net import get_embedded_net
from uGUIDE.plot_utils import plot_posterior_distribution


def estimate_microstructure(x, config, postprocessing=None, voxel_id=0, plot=True, theta_gt=None):
    """
    Estimate microstructure parameters given an observed diffusion MRI signal.
    The posterior distributions are obtained by sampling from the normalizing 
    flow. If a problem occurs, mask is set to False for the corresponding 
    parameter(s). Then the posterior distribution is defined as degenerate or not.
    Finally this function extracts and returns the maximum-a-posteriori, the
    uncertainty and the ambiguity from the estimated posterior distributions.

    Parameters
    ----------
    x : array, shape (x_size,)
        Observed diffusion MRI signal. Its size must be identical to the signals
        used for training (i.e. ``config['size_x']``)
    
    config : dict
        μGUIDE configuration.
    
    postprocessing : function, optional
        If one or multiple microstructure parameters in the model definition
        are not uniformly distributed, surrogate parameters need to be used
        instead during training. This function allows to convert those surrogate
        parameters into the model microstructure parameters. If set to None, no
        conversion is performed.
    
    voxel_id : int, default=0
        ID of the current voxel. Used to set the name when saving the posterior
        distribution plots.
    
    plot : bool, default=True
        Whether to save the posterior distributions.
    
    theta_gt : ndarray, optional
        Ground truth value corresponding to the observed signal x, whith size 
        (config['size_theta'],). Used when testing on simulations. Adds a
        vertical dashed black line on the plotted posterior distributions.

    Returns
    -------
    map : ndarray, shape (config['size_theta'],)
        Maximum-a-posteriori estimated for each microstructure parameter from
        the posterior distributions.
    
    mask : ndarray, shape (config['size_theta'],)
        Default to ``True``. Set to False if a parameter estimation did not
        work.

    degeneracy_mask : ndarray, shape (config['size_theta'],)
        Set to ``True`` if a posterior distribution is defined as degenerate.
        ``False`` otherwise.

    uncertainty : ndarray, shape (config['size_theta'],)
        Uncertainty measure estimated for each microstructure parameter from
        the posterior distributions (in %).

    ambiguity : ndarray, shape (config['size_theta'],)
        Ambiguity measure estimated for each microstructure parameter from
        the posterior distributions (in %).

    """
    samples = sample_posterior_distribution(x, config)
    if postprocessing is not None:
        samples = postprocessing(samples, config)

    map, mask, degeneracy_mask, uncertainty, ambiguity = estimate_theta(samples,
                                                                        config,
                                                                        postprocessing=postprocessing is not None)
    folderpath = config['folderpath'] / 'posterior_distributions'
    folderpath.mkdir(exist_ok=True, parents=True)

    if mask.all() == False: # If at least one is False
        if postprocessing is not None:
            param_fail = np.array(list(config["prior_postprocessing"].keys()))[mask == False]
        else:
            param_fail = np.array(list(config["prior"].keys()))[mask == False]
        print(f'Microstructure estimation of voxel {voxel_id} did not work. '\
              'Unable to fit two Gaussians on the posterior distribution of '
              f'{", ".join(param_fail)}.')

    elif plot == True:
        if postprocessing is None:
            plot_posterior_distribution(samples, config, postprocessing=False,
                                        ground_truth=theta_gt,
                                        fig_file=f'posterior_distributions/posterior_distribution_voxel_{voxel_id}.png')
            print(f'Parameters: {list(config["prior"].keys())}')
        else:
            plot_posterior_distribution(samples, config, postprocessing=True,
                                        ground_truth=theta_gt,
                                        fig_file=f'posterior_distributions/posterior_distribution_voxel_{voxel_id}.png')
            print(f'Parameters: {list(config["prior_postprocessing"].keys())}')
        
        if theta_gt is not None:
            print(f'Ground truth theta = {theta_gt}')
        
        print(f'Estimated theta = {map}')
        print(f'Degeneracies = {degeneracy_mask}')
        print(f'Uncertainties = {uncertainty} %')
        print(f'Ambiguities = {ambiguity} %')

    return map, mask, degeneracy_mask, uncertainty, ambiguity


def sample_posterior_distribution(x, config):
    # Only one observation at a time

    if x.ndim == 1:
        x = x.reshape(1,-1)

    if config['size_x'] != x.shape[1]:
        raise ValueError('x size set in config does not match x size used ' \
                         'for training')

    # Normalize data
    x_normalizer = load_normalizer(config['folderpath'] / config['x_normalizer_file'])
    x_norm = x_normalizer(x)
    x_norm = torch.from_numpy(x_norm).type(torch.float32).to(config['device'])

    nf = get_nf(input_dim=config['size_theta'],
                nf_features=config['nf_features'],
                pretrained_state=config['folderpath'] / config['nf_state_dict_file']
                )
    nf.to(config['device'])
    embedded_net = get_embedded_net(input_dim=config['size_x'],
                                    output_dim=config['nf_features'],
                                    layer_1_dim=config['hidden_layers'][0],
                                    layer_2_dim=config['hidden_layers'][1],
                                    pretrained_state=config['folderpath'] / config['embedder_state_dict_file'],
                                    use_MLP=config['use_MLP'])
    embedded_net.to(config['device'])
    embedding = embedded_net(x_norm.type(torch.float32).to(config['device']))

    # Rejection sampling
    nb_to_sample = config['nb_samples']
    loop_iter = 0
    max_loop_iter = 5000
    prior_min = np.array([config['prior'][p][0] for p in config['prior'].keys()])
    prior_max = np.array([config['prior'][p][1] for p in config['prior'].keys()])
    samples = np.zeros((nb_to_sample, config['size_theta']))
    while (nb_to_sample > 0) & (loop_iter < max_loop_iter):

        base_dist = dist.Normal(
            loc=torch.zeros((nb_to_sample,) + (config['size_theta'],)).to(config['device']),
            scale=torch.ones((nb_to_sample,) + (config['size_theta'],)).to(config['device'])
        )
        transformed_dist = dist.ConditionalTransformedDistribution(base_dist, nf)

        samples_norm = transformed_dist.condition(
                embedding
            ).sample()

        theta_normalizer = load_normalizer(config['folderpath'] / config['theta_normalizer_file'])
        candidates = theta_normalizer.inverse(samples_norm.detach().cpu().numpy())

        if (loop_iter == 100) & (len(samples) == 0):
            accepted = np.ones(accepted.shape, dtype=bool)
        elif (loop_iter < max_loop_iter):
            accepted = (candidates > prior_min).all(1) & (candidates < prior_max).all(1)
        else:
            accepted = np.ones(accepted.shape, dtype=bool)
            print(f'Nb good samples: {len(samples)}')

        if nb_to_sample == config['nb_samples']: # first iteration
            samples = candidates[accepted]
        else:
            samples = np.append(samples, candidates[accepted], axis=0)

        nb_to_sample = config['nb_samples'] - len(samples)
        loop_iter += 1

    return samples


def estimate_theta(samples, config, postprocessing=False):
    # Get MAP, degeneracy, uncertainty and ambiguity

    # Check if samples have the save size as size_theta in config
    if (postprocessing == False) & (config['size_theta'] != samples.shape[1]):
            raise ValueError('Theta size set in config does not match theta ' \
                            'size used for training')
    elif (postprocessing == True) & (len(config['prior_postprocessing']) != samples.shape[1]):
            raise ValueError('Theta size does not match theta size of postprocessing.')

    if postprocessing == True:
        prior = config['prior_postprocessing']
    else:
        prior = config['prior']

    theta_mean = samples.mean(0)

    map = theta_mean
    mask = np.ones(len(prior), dtype=bool)
    degeneracy_mask = np.zeros(len(prior), dtype=bool)
    uncertainty = np.ones(len(prior)) * 100
    ambiguity = np.ones(len(prior)) * 100

    for i, param in enumerate(prior.keys()):
        if (theta_mean[i] < prior[param][0]) \
            or (theta_mean[i] > prior[param][1]):
            mask[i] = False
        else:
            # Only compute degeneracy for non-masked/valid voxel estimations
            x_hist, hist = get_hist(samples[:,i])
            param_gauss = fit_two_gaussians(x_hist, hist)
            # If the gaussian fitting did not work, set this voxel's parameter as invalid
            if np.all(param_gauss == 0):
                mask[i] = False
            else:
                degeneracy_mask[i] = is_degenerate(param_gauss, prior[param])
                map[i] = estimate_max_a_posteriori(param_gauss, prior[param])
                if degeneracy_mask[i] == False: # If degenerate, uncertainty and ambiguity are set to 100%
                    ambiguity[i] = estimate_ambiguity(param_gauss, prior[param])
                    uncertainty[i] = estimate_uncertainty(samples[:,i], prior[param])

    return map, mask, degeneracy_mask, uncertainty, ambiguity


def is_degenerate(param_gauss, prior_bounds):
    degenerate = False

    x = np.linspace(prior_bounds[0], prior_bounds[1], 1000)
    der = derivative_two_gaussians(x, param_gauss[0], param_gauss[1], param_gauss[2], param_gauss[3], param_gauss[4], param_gauss[5])

    sign_d = sign_der(der)
    idx_der = np.where(sign_d[:-1] != sign_d[1:])[0] + 1

    # If idx_der contain two consecutive numbers, means it is a suprious spike.
    # Do not take it into account
    if len(idx_der) > 1:
        der_to_keep = []
        for i_d, d in enumerate(idx_der):
            before = False
            after = False
            if i_d != 0: # if not first one
                if d - 1 == idx_der[i_d-1]: # consecutive with the one before
                    before = True
            if i_d != len(idx_der) - 1: # not the last one  
                if d + 1 == idx_der[i_d+1]: # consecutive with the one after
                    after = True
            if after == False and before == False:
                der_to_keep.append(d)
        idx_der = der_to_keep

    if len(idx_der) > 1:
        # If distance between the mean of the two gaussians < sum of std,
        # then the two are too close to distinguish them. Not degenerate
        dist_mean = np.abs(param_gauss[1] - param_gauss[4])
        if dist_mean > param_gauss[2] + param_gauss[5]:
            degenerate = True

    return degenerate


def estimate_max_a_posteriori(param_gauss, prior_bounds):
    x = np.linspace(prior_bounds[0], prior_bounds[1], 1000)
    y = two_gaussians(x, param_gauss[0], param_gauss[1], param_gauss[2], param_gauss[3], param_gauss[4], param_gauss[5])
    map = x[y == y.max()]
    return map
    

def estimate_ambiguity(param_gauss, prior_bounds):
    x = np.linspace(prior_bounds[0], prior_bounds[1], 1000)
    y = two_gaussians(x, param_gauss[0], param_gauss[1], param_gauss[2], param_gauss[3], param_gauss[4], param_gauss[5])
    ambiguity = (len(np.where(y > y.max()/2)[0]) / x.shape[0]) * 100
    return ambiguity


def estimate_uncertainty(samples, prior_bounds):
    q3, q1 = np.percentile(samples, [75, 25])
    iqr = q3 - q1
    uncertainty = iqr / (prior_bounds[1] - prior_bounds[0])
    uncertainty *= 100
    return uncertainty


def one_gaussian(x, f, mu, sigma):
    return f * 1/(sigma*(np.sqrt(2*np.pi))) * np.exp(-1/2 * ((x - mu) / sigma)**2)


def two_gaussians(x, f1, mu1, sigma1, f2, mu2, sigma2):
    return one_gaussian(x, f1, mu1, sigma1) + one_gaussian(x, f2, mu2, sigma2)


def derivative_one_gaussian(x, f, mu, sigma):
    return - (x - mu)/sigma**2 * one_gaussian(x, f, mu, sigma)


def derivative_two_gaussians(x, f1, mu1, sigma1, f2, mu2, sigma2):
    return derivative_one_gaussian(x, f1, mu1, sigma1) + derivative_one_gaussian(x, f2, mu2, sigma2)


def sign_der(derivative):
    s = np.sign(derivative)
    if s[0] == 0:
        s[0] = s[s != 0][0]
    for i in np.arange(1, s.shape[0]):
        if s[i] == 0:
            s[i] = s[i-1]
    return s


def get_hist(samples):
    hist, bin_edges = np.histogram(samples, density=True, bins=100)
    n = len(hist)
    x_hist=np.zeros((n),dtype=float) 
    for ii in range(n):
        x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
    return x_hist, hist


def fit_two_gaussians(x_hist, hist):
    min_hist = x_hist[0]
    max_hist = x_hist[-1]
    try:
        param_gauss, _ = optimize.curve_fit(two_gaussians, x_hist, hist,
                                            bounds=([0.0, min_hist, 0.0, 0.0, min_hist, 0.0], [10, max_hist, max_hist, 10, max_hist, max_hist]))
    except Exception:
        param_gauss = np.zeros(6)
        # print(traceback.format_exc())
    return param_gauss
