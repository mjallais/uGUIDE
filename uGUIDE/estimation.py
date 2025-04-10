import traceback
import numpy as np
import torch
import pyro.distributions as dist
from scipy import optimize
# import traceback
import matplotlib.pyplot as plt
from numba import jit
import torch.nn.functional as F
from scipy.signal import find_peaks
import torch.nn as nn


from uGUIDE.plot_utils import plot_posterior_distribution


def estimate_microstructure(x, config, nf, embedded_net, x_normalizer,
                            theta_normalizer, postprocessing=None, voxel_id=0,
                            plot=True, theta_gt=None):
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
    print(f'Voxel {voxel_id}')
    samples = sample_posterior_distribution(x, config, nf, embedded_net,
                                            x_normalizer, theta_normalizer)
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

    if plot == True:
        if postprocessing is None:
            plot_posterior_distribution(samples, config, postprocessing=False,
                                        ground_truth=theta_gt,
                                        fig_file=f'posterior_distributions/posterior_distribution_voxel_{voxel_id}.png')
            print(f'Parameters: {list(config["prior"].keys())}')
        else:
            plot_posterior_distribution(samples, config, postprocessing=True,
                                        ground_truth=theta_gt,
                                        fig_file=f'posterior_distributions/posterior_distribution_voxel_{voxel_id}_postprocessing.png')
            print(f'Parameters: {list(config["prior_postprocessing"].keys())}')
        
        if theta_gt is not None:
            print(f'Ground truth theta = {theta_gt}')
        
        print(f'Estimated theta = {map}')
        print(f'Degeneracies = {degeneracy_mask}')
        print(f'Uncertainties = {uncertainty} %')
        print(f'Ambiguities = {ambiguity} %')

    return map, mask, degeneracy_mask, uncertainty, ambiguity


def sample_posterior_distribution(x, config, nf, embedded_net, x_normalizer, theta_normalizer):
    # Batch processing (Multiple voxels at once)
    
    if x.ndim == 1:
        x = x.reshape(1,-1)

    if config['size_x'] != x.shape[1]:
        raise ValueError('x size set in config does not match x size used ' \
                         'for training')

    # Normalize data
    # x_norm = torch.tensor(x_normalizer(x), dtype=torch.float32, device=config['device'])
    x_norm = x_normalizer(x).to(config['device'])
    embedding = embedded_net(x_norm)

    # Rejection sampling
    nb_to_sample = config['nb_samples']
    loop_iter = 0
    max_loop_iter = 5000
    prior_min = torch.Tensor([config['prior'][p][0] for p in config['prior'].keys()],
                             device=config['device'])
    prior_max = torch.Tensor([config['prior'][p][1] for p in config['prior'].keys()],
                             device=config['device'])
    samples = torch.tensor([], device=config['device'])
    while (nb_to_sample > 0) and (loop_iter < max_loop_iter):

        base_dist = dist.Normal(
            loc=torch.zeros((nb_to_sample,) + (config['size_theta'],), device=config['device']),
            scale=torch.ones((nb_to_sample,) + (config['size_theta'],), device=config['device'])
        )
        transformed_dist = dist.ConditionalTransformedDistribution(base_dist, nf)

        with torch.no_grad():
            samples_norm = transformed_dist.condition(embedding).sample()
            candidates = theta_normalizer.inverse(samples_norm)

        # if loop_iter == 100 and samples.size == 0:
        #     accepted = np.ones_like(candidates[:, 0], dtype=bool)
        # elif (loop_iter < max_loop_iter):
        #     accepted = np.all((candidates > prior_min) & (candidates < prior_max), axis=1)
        # else:
        #     accepted = np.ones(accepted.shape, dtype=bool)
        #     print(f'Nb good samples: {len(samples)}')

        if loop_iter == 100 and len(samples) == 0:
            accepted = torch.ones(accepted.shape, dtype=bool, device=config['device'])
        else:
            accepted = torch.all((candidates > prior_min) & (candidates < prior_max), axis=1)
        samples = torch.cat((samples, candidates[accepted]), dim=0)

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
    # mask = np.ones(len(prior), dtype=bool)
    mask = (theta_mean >= torch.tensor([prior[param][0] for param in prior.keys()])) & \
           (theta_mean <= torch.tensor([prior[param][1] for param in prior.keys()]))
    print(f'mask = {mask}')

    degeneracy_mask = torch.zeros(len(prior), dtype=torch.bool, device=config['device'])
    uncertainty = torch.ones(len(prior), dtype=torch.float32, device=config['device']) * 100
    ambiguity = torch.ones(len(prior), dtype=torch.float32, device=config['device']) * 100

    # For testing 
    if postprocessing == False:
        prior = config['prior']
    else:
        prior = config['prior_postprocessing']
    
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(prior),
        figsize=(5 * len(prior), 5),
        sharey="row"
    )
    for p, param in enumerate(prior.keys()):
        hist_uGUIDE, bin_edges = torch.histogram(samples[:,p], density=True,
                                                 bins=100)
        x_hist_uGUIDE = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins


        axs[p].plot(x_hist_uGUIDE, hist_uGUIDE)
        axs[p].fill_between(x_hist_uGUIDE, hist_uGUIDE, alpha=0.4)
        axs[p].set_xlabel(param, fontsize=20)
        axs[p].set_xlim(prior[param][0], prior[param][1])
        axs[p].set_yticks([])
        axs[p].tick_params(axis='x', which='major', labelsize=20)

    for i, param in enumerate(prior.keys()):
        # if (theta_mean[i] < prior[param][0]) \
        #     or (theta_mean[i] > prior[param][1]):
        #     mask[i] = False
        # else:
        if mask[i]:
            # Only compute degeneracy for non-masked/valid voxel estimations
            # x_hist, hist = get_hist(samples[:,i], prior[param])
            hist, bin_edges = torch.histogram(samples[:,i], density=True,
                                              bins=100)
            x_hist = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins
            # param_gauss = fit_two_gaussians(x_hist, hist, config)
            # print(f'samples mean = {samples[:,i].mean()}')
            param_gauss = fit_two_gaussians(samples[:,i], config, prior[param])
            # print(f'param_gauss = {param_gauss}')
            # print(f'two gaussians = {two_gaussians(x_hist, *param_gauss)}')
            axs[i].plot(x_hist, two_gaussians(x_hist, *param_gauss).detach().numpy(),
                        color='red', label='Fitted')
            # print(f'mu1 = {param_gauss[1]}, mu2 = {param_gauss[4]}')
            # param_gauss = torch.Tensor(fit_two_gaussians(x_hist.detach().numpy(), hist.detach().numpy()),
            #                            device=config['device'])
            # If the gaussian fitting did not work, set this voxel's parameter as invalid
            if param_gauss[0] == 0 and param_gauss[3] == 0:
                mask[i] = False
            else:
                degeneracy_mask[i] = is_degenerate(param_gauss, prior[param], config)
                map[i] = estimate_max_a_posteriori(param_gauss, prior[param], config)
                if degeneracy_mask[i] == False: # If degenerate, uncertainty and ambiguity are set to 100%
                    ambiguity[i] = estimate_ambiguity(param_gauss, prior[param], config)
                    uncertainty[i] = estimate_uncertainty(samples[:,i], prior[param])


    fig.tight_layout()
    plt.show()
    print(f'Degeneracy mask = {degeneracy_mask}')



    return map, mask, degeneracy_mask, uncertainty, ambiguity


def is_degenerate(param_gauss, prior_bounds, config, num_points=1000):
    """
    Determines degeneracy of a Gaussian mixture by checking for multiple significant peaks.
    Vectorized with PyTorch.

    Parameters:
        param_gauss (torch.Tensor): Tensor of Gaussian parameters, shape (num_params, 6).
        prior_bounds (torch.Tensor): Tensor of min/max bounds, shape (num_params, 2).
        num_points (int): Number of points for evaluating the Gaussians.

    Returns:
        torch.Tensor: Boolean tensor indicating degeneracy for each parameter, shape (num_params,).
    """
    degenerate = False

    x = torch.linspace(prior_bounds[0], prior_bounds[1], num_points,
                       device=config['device'])
    der = derivative_two_gaussians(x, *param_gauss) # check if same device as x
# !!!!!!!!!!!!!!!!!!!!!!!!!

    sign_d = sign_der(der)
    idx_der = np.where(sign_d[:-1] != sign_d[1:])[0] + 1

    # idx_der = torch.sign(der[:-1]) != torch.sign(der[1:])  # Zero-crossings

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
        dist_mean = torch.abs(param_gauss[1] - param_gauss[4])
        if dist_mean > param_gauss[2] + param_gauss[5]:
            degenerate = True

    return degenerate


def estimate_max_a_posteriori(param_gauss, prior_bounds, config, num_points=1000):
    x = torch.linspace(prior_bounds[0], prior_bounds[1], num_points,
                       device=config['device'])
    y = two_gaussians(x, *param_gauss)
    map = x[y == y.max()]
    return map


def estimate_ambiguity(param_gauss, prior_bounds, config, num_points=1000):
    """
    Computes ambiguity as the percentage of the domain where the posterior is above half-max.
    Vectorized with PyTorch.
    
    Parameters:
        param_gauss (torch.Tensor): Tensor of Gaussian parameters, shape (num_params, 6).
        prior_bounds (torch.Tensor): Tensor of min/max bounds, shape (num_params, 2).
        num_points (int): Number of points for evaluating the Gaussians.

    Returns:
        torch.Tensor: Ambiguity measure for each parameter, shape (num_params,).
    """
    # x = np.linspace(prior_bounds[0], prior_bounds[1], num_points)
    # 
    # # ambiguity = (len(np.where(y > y.max()/2)[0]) / x.shape[0]) * 100
    # # return ambiguity
    # return (np.count_nonzero(y > y.max() / 2) / x.shape[0]) * 100

    x = torch.linspace(prior_bounds[0], prior_bounds[1], num_points,
                       device=config['device'])
    y = two_gaussians(x, *param_gauss)

    half_max = y.max(dim=0, keepdim=True)[0] / 2
    ambiguity = (torch.sum(y > half_max, dim=0).float() / num_points) * 100  # Percentage of domain above half-max
    return ambiguity


def estimate_uncertainty(samples, prior_bounds):
    """
    Computes uncertainty using interquartile range (IQR) as a percentage of the prior range.
    Vectorized with PyTorch for GPU acceleration.
    
    Parameters:
        samples (torch.Tensor): Tensor of posterior samples, shape (N, num_params).
        prior_bounds (torch.Tensor): Tensor of shape (num_params, 2) with min/max bounds.

    Returns:
        torch.Tensor: Uncertainty for each parameter in percentage, shape (num_params,).
    """
    # q3, q1 = np.percentile(samples, [75, 25])
    q1 = torch.quantile(samples, 0.25, dim=0)
    q3 = torch.quantile(samples, 0.75, dim=0)
    iqr = q3 - q1
    # uncertainty = iqr / (prior_bounds[1] - prior_bounds[0])
    # uncertainty *= 100
    # return uncertainty
    return (iqr / (prior_bounds[1] - prior_bounds[0])) * 100


def one_gaussian(x, f, mu, sigma):
    SQRT_2PI = torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=torch.float32, device=x.device))
    return f * 1/(sigma*SQRT_2PI) * torch.exp(-1/2 * ((x - mu) / sigma)**2)


def two_gaussians(x, f1, mu1, sigma1, f2, mu2, sigma2):
    return one_gaussian(x, f1, mu1, sigma1) + one_gaussian(x, f2, mu2, sigma2)


def derivative_one_gaussian(x, f, mu, sigma):
    return - (x - mu)/sigma**2 * one_gaussian(x, f, mu, sigma)


def derivative_two_gaussians(x, f1, mu1, sigma1, f2, mu2, sigma2):
    return derivative_one_gaussian(x, f1, mu1, sigma1) + derivative_one_gaussian(x, f2, mu2, sigma2)


def sign_der(derivative):
    s = torch.sign(derivative)
    # s = np.sign(derivative)
    if s[0] == 0:
        s[0] = s[s != 0][0]
    for i in np.arange(1, s.shape[0]):
        if s[i] == 0:
            s[i] = s[i-1]
    return s


def get_hist(samples):
    hist, bin_edges = torch.histogram(samples, density=False, bins=100)
    # n = len(hist)
    # x_hist=np.zeros((n),dtype=float) 
    # for ii in range(n):
    #     x_hist[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
    x_hist = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of bins
    return x_hist, hist


# def fit_two_gaussians(x_hist, hist, config, lr=1e-2, max_iter=500):
#     min_hist = x_hist[0]
#     max_hist = x_hist[-1]

#     # Initialize parameters: [f1, mu1, sigma1, f2, mu2, sigma2]
#     params = torch.nn.Parameter(torch.tensor([
#         1.0,                              # f1
#         (min_hist + max_hist) / 3,        # mu1
#         x_hist.std(),                     # sigma1
#         1.0,                              # f2
#         2 *(min_hist + max_hist) / 3,     # mu2
#         x_hist.std()                      # sigma2
#     ], dtype=torch.float32, device=config['device']))

#     # optimizer = torch.optim.SGD([params], lr=lr, momentum=0.9)
#     optimizer = torch.optim.LBFGS([params], max_iter=max_iter, line_search_fn='strong_wolfe')

#     # lower_bounds = torch.tensor([0.0, min_hist, 1e-3, 0.0, min_hist, 1e-3],
#     #                             device=config['device'])
#     # upper_bounds = torch.tensor([10.0, max_hist, max_hist, 10.0, max_hist, max_hist],
#     #                             device=config['device'])

#     # def closure():
#     #     optimizer.zero_grad()
#     #     # clamped_params = torch.max(torch.min(params, upper_bounds), lower_bounds)
#     #     # clamped_params = params
#     #     y_pred = two_gaussians(x_hist, *params)  # Unclamped params
#     #                         #    clamped_params[0], clamped_params[1],
#     #                         #    clamped_params[2], clamped_params[3],
#     #                         #    clamped_params[4], clamped_params[5])        
#     #     loss = torch.mean((hist - y_pred)**2)
#     #     # separation_penalty = -torch.abs(params[1] - params[4])  # mu1 - mu2
#     #     # loss = loss + 0.1 * separation_penalty
#     #     loss.backward()
#     #     return loss

#     amplitude_upper = hist.max() * 1.5  # 50% buffer above max height
    
#     def transform(raw, amplitude_upper=amplitude_upper):
#         lower = torch.tensor([0.0, min_hist, 1e-4,
#                               0.0, min_hist, 1e-4],
#                               device=config['device'])
#         upper = torch.tensor([amplitude_upper, max_hist, (max_hist-min_hist)/2, 
#                               amplitude_upper, max_hist, (max_hist-min_hist)/2],
#                               device=config['device'])
#         return lower + (upper - lower) * torch.sigmoid(raw)
    
#     optimizer = torch.optim.Adam([params], lr=lr)
#     for i in range(max_iter):
#         optimizer.zero_grad()
#         params = transform(params)
#         y_pred = two_gaussians(x_hist, *params)
#         loss = torch.mean((hist - y_pred) ** 2)
#         loss.backward()
#         optimizer.step()

#     # try:
#     #     # for _ in range(max_iter):
#     #         # optimizer.zero_grad()

#     #         # Clamp parameters within bounds
#     #         # clamped_params = torch.max(torch.min(params, upper_bounds), lower_bounds)
#     #         # clamped_params = torch.clamp(params, lower_bounds, upper_bounds)
#     #         # clamped_params = params
#     #         # y_pred = two_gaussians(x_hist,
#     #         #                        clamped_params[0], clamped_params[1],
#     #         #                        clamped_params[2], clamped_params[3],
#     #         #                        clamped_params[4], clamped_params[5])

#     #         # loss = torch.mean((hist - y_pred)**2)
#     #         # loss.backward()
#     #         # optimizer.step()
#     #     optimizer.step(closure)

#     #     # fitted = torch.max(torch.min(params, upper_bounds), lower_bounds)
#     #     # fitted = torch.clamp(params, lower_bounds, upper_bounds)
#     #     fitted = params

#     # except Exception:
#     #     fitted = torch.zeros(6)
#     #     print(traceback.format_exc())
    
#     fitted=params
#     return fitted.to(config['device'])

#     # try:
#     #     param_gauss, _ = optimize.curve_fit(two_gaussians, x_hist, hist,
#     #                                         bounds=([0.0, min_hist, 0.0, 0.0, min_hist, 0.0], [10, max_hist, max_hist, 10, max_hist, max_hist]),
#     #     )
#     # except Exception:
#     #     print('Gaussian fitting did not work.')
#     #     print(traceback.format_exc())
#     #     param_gauss = np.zeros(6)
#     #     # print(traceback.format_exc())
#     # return param_gauss

# def get_transform_bounds(hist, min_hist, max_hist, device):
#     amplitude_upper = hist.max() * 1.5
#     hist_range = max_hist - min_hist
#     lower = torch.tensor([0.0, min_hist, 1e-4, 0.0, min_hist, 1e-4], device=device)
#     upper = torch.tensor([
#         amplitude_upper, max_hist, hist_range / 2,
#         amplitude_upper, max_hist, hist_range / 2
#     ], device=device)
#     return lower, upper

# def transform(raw, lower, upper):
#     return lower + (upper - lower) * torch.sigmoid(raw)

# def fit_two_gaussians(x_hist, hist, config, lr=1e-2, max_iter=1000):
#     min_hist = x_hist.min()
#     max_hist = x_hist.max()

#     lower, upper = get_transform_bounds(hist, min_hist, max_hist, config['device'])

#     # raw_params = torch.nn.Parameter(torch.randn(6, device=device))  # Initial unbounded params
#     # Initialize parameters: [f1, mu1, sigma1, f2, mu2, sigma2]
#     raw_params = torch.nn.Parameter(torch.tensor([
#         1.0,                              # f1
#         (min_hist + max_hist) / 3,        # mu1
#         x_hist.std(),                     # sigma1
#         1.0,                              # f2
#         2 *(min_hist + max_hist) / 3,     # mu2
#         x_hist.std()                      # sigma2
#     ], dtype=torch.float32, device=config['device']))
#     optimizer = torch.optim.Adam([raw_params], lr=lr)

#     lambda_sep_penaly = 0.1  # Separation penalty weight
#     lambda_f_penaly = 0  # Separation penalty weight
#     lambda_std_penaly = 0  # Separation penalty weight

    
#     MSE_loss = []
#     mean_sep_penalty_loss = []
#     mean_f_penalty_loss = []
#     mean_std_penalty_loss = []
    
#     n_restarts=20
#     best_loss = float('inf')
#     best_params = None

#     for _ in range(n_restarts):
#         # Raw parameters: [amp1, mu1, raw_std1, amp2, mu2, raw_std2]
#         raw_params = torch.nn.Parameter(torch.randn(6, device=config['device']))
#         optimizer = torch.optim.Adam([raw_params], lr=lr)

#         for i in range(max_iter):
#             optimizer.zero_grad()
#             f1 = raw_params[0]
#             # f1 = torch.sigmoid(raw_params[0]) * (amplitude_upper - amplitude_lower) + amplitude_lower
#             mu1 = torch.sigmoid(raw_params[1]) * (max_hist - min_hist) + min_hist
#             # mu1 = F.softplus(raw_params[1])
#             sigma1 = F.softplus(raw_params[2]) + 1e-3
#             f2 = raw_params[3]
#             # f2 = torch.sigmoid(raw_params[3]) * (amplitude_upper - amplitude_lower) + amplitude_lower
#             mu2 = torch.sigmoid(raw_params[4]) * (max_hist - min_hist) + min_hist
#             # mu2 = F.softplus(raw_params[4])
#             sigma2 = F.softplus(raw_params[5]) + 1e-3
#             # params = transform(raw_params, lower, upper)
#             y_pred = two_gaussians(x_hist, f1, mu1, sigma1, f2, mu2, sigma2)
#             mean_sep_penalty = torch.exp(-torch.abs(mu1 - mu2))  # or 1 / (1 + |mu1 - mu2|)
#             mean_f_penalty = torch.abs(f1 - f2)
#             mean_std_penalty = torch.abs(sigma1 - sigma2)
#             loss = F.mse_loss(y_pred, hist) \
#                     + lambda_sep_penaly * mean_sep_penalty \
#                     + lambda_f_penaly * mean_f_penalty \
#                     + lambda_std_penaly * mean_std_penalty
#             if i % 50 == 0:
#                 MSE_loss.append(loss.item())
#                 mean_sep_penalty_loss.append(mean_sep_penalty.item())
#                 mean_f_penalty_loss.append(mean_f_penalty.item())
#                 mean_std_penalty_loss.append(mean_std_penalty.item())
#             loss.backward()
#             optimizer.step()

#         if loss.item() < best_loss:
#             best_loss = loss.item()
#             best_params = torch.tensor([f1, mu1, sigma1, f2, mu2, sigma2], device=config['device'])


#     print(f'MSE loss = {MSE_loss}')
#     print(f'Mean separation penalty = {mean_sep_penalty_loss}')
#     print(f'Mean f penalty = {mean_f_penalty_loss}')
#     print(f'Mean std penalty = {mean_std_penalty_loss}')
#     # fitted = transform(raw_params, lower, upper)
#     # return torch.Tensor([f1, mu1, sigma1, f2, mu2, sigma2], device=config['device'])
#     return best_params

class GMM2(nn.Module):
    def __init__(self, x, device='cpu'):
        super().__init__()
        self.device = device
        self.x = x.to(device)

        # Logits for mixture weights
        self.logits = nn.Parameter(torch.tensor([0.0, 0.0], device=device))  # Softmaxed later

        # Means and stddevs
        self.raw_means = nn.Parameter(torch.tensor([x.mean()*0.9, x.mean()*1.1], device=device))
        self.raw_log_stds = nn.Parameter(torch.tensor([-1.0, -1.0], device=device))  # small stds
    
    def forward(self, prior):
        # Convert parameters
        weights = F.softmax(self.logits, dim=0).clamp(min=1e-3, max=1.0)

        # Sigmoid squashes raw_means to (0,1), then scale to prior range
        mu1 = torch.sigmoid(self.raw_means[0]) * (prior[1] - prior[0]) + prior[0]
        mu2 = torch.sigmoid(self.raw_means[1]) * (prior[1] - prior[0]) + prior[0]

        # stds = softplus(log_std) + small constant
        sigma1 = F.softplus(self.raw_log_stds[0]) + 1e-3
        sigma2 = F.softplus(self.raw_log_stds[1]) + 1e-3

        x = self.x

        mixture = two_gaussians(x, weights[0], mu1, sigma1,
                                weights[1], mu2, sigma2)

        # Return negative log likelihood
        return -torch.log(mixture + 1e-12).mean()
    
def fit_two_gaussians(samples, config, prior, lr=1e-2, max_iter=500):
    model = GMM2(samples, device=config['device'])
    model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(max_iter):
        loss = model(prior=prior)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            print(f"Iter {i}: Loss = {loss.item():.4f}")
    
    weights = F.softmax(model.logits, dim=0)
    mu1 = torch.sigmoid(model.raw_means[0]) * (prior[1] - prior[0]) + prior[0]
    mu2 = torch.sigmoid(model.raw_means[1]) * (prior[1] - prior[0]) + prior[0]
    sigma1 = F.softplus(model.raw_log_stds[0]) + 1e-3
    sigma2 = F.softplus(model.raw_log_stds[1]) + 1e-3

    return torch.Tensor([weights[0], mu1, sigma1,
                         weights[1], mu2, sigma2],
                        device=config['device'])