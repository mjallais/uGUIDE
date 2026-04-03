from typing import Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.distributions as dist
from scipy import optimize
from time import time
from tqdm import tqdm

from uGUIDE.plot_utils import plot_posterior_distribution
from uGUIDE.density_estimator import get_nf
from uGUIDE.embedded_net import get_embedded_net
from uGUIDE.normalization import load_normalizer, load_bounded_normalizer


def estimate_microstructure(
    x: torch.Tensor,
    config: Dict,
    postprocessing=None,
    verbose: bool = False,
    plot: bool = False,
    plot_max_voxels: int = 5,
    plot_random: bool = True,
    theta_gt: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """
    Estimate microstructure parameters given observed diffusion MRI signals.
    The posterior distributions are obtained by sampling from the normalizing 
    flow. If a problem occurs, mask is set to False for the corresponding 
    parameter(s). Then the posterior distribution is defined as degenerate or
    not. Finally this function extracts and returns the maximum-a-posteriori, 
    the uncertainty and the ambiguity from the estimated posterior
    distributions for each voxel.

    Parameters
    ----------
    x : torch.Tensor, shape (N_voxels, config['size_x'])
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

    verbose : bool, default=False
        Whether to print messages about the estimation process, especially 
        invalid cases where the estimation did not work.
    
    plot : bool, default=True
        Whether to save the posterior distributions.

    plot_max_voxels : int, default=5
        Maximum number of voxels to plot the posterior distributions for. If the
        number of voxels to estimate is higher than this number, only a random
        subset of voxels will be plotted if plot_random is set to True. If
        plot_random is set to False, the first plot_max_voxels will be plotted.
    
    plot_random : bool, default=True
        Whether to randomly select the voxels to plot the posterior distributions
        for when the number of voxels to estimate is higher than plot_max_voxels.
    
    theta_gt : torch.Tensor, optional
        Ground truth value corresponding to the observed signal x, whith size 
        (config['size_theta'],). Used when testing on simulations. Adds a
        vertical dashed black line on the plotted posterior distributions.

    Returns
    -------
    map : torch.Tensor, shape (N, size_theta)
        Maximum-a-posteriori estimated for each microstructure parameter from
        the posterior distributions.
    
    mask : torch.Tensor, shape (N, size_theta)
        Default to ``True``. Set to False if a parameter estimation did not
        work.

    degeneracy_mask : torch.Tensor, shape (N, size_theta)
        Set to ``True`` if a posterior distribution is defined as degenerate.
        ``False`` otherwise.

    uncertainty : torch.Tensor, shape (N, size_theta)
        Uncertainty measure estimated for each microstructure parameter from
        the posterior distributions (in %).

    ambiguity : torch.Tensor, shape (N, size_theta)
        Ambiguity measure estimated for each microstructure parameter from
        the posterior distributions (in %).

    """
    torch.set_num_threads(config['num_threads'])

    device = config['device']
    x = x.to(device)

    N = x.shape[
        0]  # Number of voxels to estimate microstructure parameters for
    D = config['size_theta']  # Number of microstructure parameters to estimate
    S = config[
        'nb_samples']  # Number of samples to draw from the posterior distribution for each voxel

    # Load normalizers
    theta_normalizer = load_bounded_normalizer(config['folderpath'] /
                                               config['theta_normalizer_file'])
    theta_normalizer = theta_normalizer.to(device)
    x_normalizer = load_normalizer(config['folderpath'] /
                                   config['x_normalizer_file'])
    x_normalizer = x_normalizer.to(device)

    # Load trained model
    embedded_net = get_embedded_net(input_dim=config['size_x'],
                                    output_dim=config['nf_features'],
                                    layer_1_dim=config['hidden_layers'][0],
                                    layer_2_dim=config['hidden_layers'][1],
                                    pretrained_state=config['folderpath'] /
                                    config['embedder_state_dict_file'],
                                    use_MLP=config['use_MLP']).to(device)
    nf = get_nf(input_dim=config['size_theta'],
                nf_features=config['nf_features'],
                n_flows=config['n_flows'],
                pretrained_state=config['folderpath'] /
                config['nf_state_dict_file']).to(device)

    embedded_net.eval()
    nf.eval()

    # Process voxels in batches to manage memory usage
    voxel_batch_size = config.get("voxel_batch_size", 256)

    results = {
        "map": [],
        "mask": [],
        "degeneracy": [],
        "uncertainty": [],
        "ambiguity": []
    }

    start_time = time()

    pbar = tqdm(total=N, desc="Estimating microstructure parameters")
    for v_start in range(0, N, voxel_batch_size):

        v_end = min(v_start + voxel_batch_size, N)

        x_batch = x[v_start:v_end]
        B = x_batch.shape[0]  # Actual batch size (last batch might be smaller)

        with torch.inference_mode():
            x_norm = x_normalizer(x_batch)

            embedding = embedded_net(x_norm)
            embedding = torch.tanh(embedding)
            embedding = F.layer_norm(embedding, embedding.shape[-1:])

            # Sample posterior: one shot sampling from the normalizing flow in the normalized space
            base_dist = dist.Normal(
                torch.zeros((B, D), device=device),
                torch.ones((B, D), device=device),
            ).to_event(1)
            transformed_dist = dist.ConditionalTransformedDistribution(
                base_dist, nf)
            cond_dist = transformed_dist.condition(embedding)

            theta_norm_samples = cond_dist.sample((S, ))  # (S, B, D)
            theta_norm_samples = theta_norm_samples.permute(1, 0,
                                                            2)  # (B, S, D)

            # Inverse normalization
            theta_samples = theta_normalizer.inverse(
                theta_norm_samples)  # (B, S, D)

            if theta_samples.ndim == 2:  # (S, D) - for single voxel, add batch dimension
                theta_samples = theta_samples.unsqueeze(0)

        if postprocessing is not None:
            B = theta_samples.shape[0]
            theta_samples = postprocessing(theta_samples.reshape(-1, D),
                                           config)
            theta_samples = theta_samples.view(B, S, -1)

        # Estimate MAP, degeneracy, uncertainty and ambiguity
        map_b, mask_b, degeneracy_mask_b, uncertainty_b, ambiguity_b = estimate_theta(
            theta_samples, config, postprocessing is not None)

        results["map"].append(map_b.cpu())
        results["mask"].append(mask_b.cpu())
        results["degeneracy"].append(degeneracy_mask_b.cpu())
        results["uncertainty"].append(uncertainty_b.cpu())
        results["ambiguity"].append(ambiguity_b.cpu())
        pbar.update(B)

    pbar.close()

    # Concatenate results from all batches
    map = torch.cat(results["map"], dim=0)
    mask = torch.cat(results["mask"], dim=0)
    degeneracy_mask = torch.cat(results["degeneracy"], dim=0)
    uncertainty = torch.cat(results["uncertainty"], dim=0)
    ambiguity = torch.cat(results["ambiguity"], dim=0)

    end_time = time()

    # Verbose logging
    if verbose:
        computation_time = end_time - start_time  # in seconds

        keys = (list(config["prior_postprocessing"].keys())
                if postprocessing else list(config["prior"].keys()))

        failed_voxels = (~mask).any(dim=-1)
        n_failed = failed_voxels.sum().item()

        print(f"\n=== μGUIDE Inference Summary ===")
        print(f"Total voxels: {N}")
        print(f"Computation time: {int(computation_time // 3600)}h "
              f"{int((computation_time % 3600) // 60)}min "
              f"{computation_time % 60:.2f}s")
        print(f"Failed voxels: {n_failed} ({100*n_failed/N:.2f}%)")
        print(f"Degenerate voxels: {degeneracy_mask.any(dim=-1).sum().item()} "
              f"({100*degeneracy_mask.any(dim=-1).sum().item()/N:.2f}%)")

        if n_failed > 0:
            print("\n--- Failed voxel details (first 10) ---")

            for n in torch.where(failed_voxels)[0][:10]:  # limit output
                param_fail = [k for k, m in zip(keys, mask[n].cpu()) if not m]
                print(
                    f"Voxel {n.item()} failed for parameters {', '.join(param_fail)}"
                )

    # Plotting posterior distributions for a subset of voxels
    if plot:

        # Choose voxels to plot
        if plot_random:
            import random  # Import here to avoid unnecessary dependency if not plotting
            idx = random.sample(range(N), min(plot_max_voxels, N))
        else:
            idx = list(range(min(plot_max_voxels, N)))

        x_idx = x[idx]
        B = x_idx.shape[0]

        with torch.inference_mode():
            x_norm = x_normalizer(x_idx)

            embedding = embedded_net(x_norm)
            embedding = torch.tanh(embedding)
            embedding = F.layer_norm(embedding, embedding.shape[-1:])

            # Sample posterior: one shot sampling from the normalizing flow in the normalized space
            base_dist = dist.Normal(
                torch.zeros((B, D), device=device),
                torch.ones((B, D), device=device),
            ).to_event(1)
            transformed_dist = dist.ConditionalTransformedDistribution(
                base_dist, nf)
            cond_dist = transformed_dist.condition(embedding)

            theta_norm_samples = cond_dist.sample((S, ))  # (S, B, D)
            theta_norm_samples = theta_norm_samples.permute(1, 0,
                                                            2)  # (B, S, D)

            # Inverse normalization
            theta_samples = theta_normalizer.inverse(
                theta_norm_samples)  # (B, S, D)

            if theta_samples.ndim == 2:  # (S, D) - for single voxel, add batch dimension
                theta_samples = theta_samples.unsqueeze(0)

        if theta_samples.ndim == 2:  # (S, D) - for single voxel, add batch dimension
            theta_samples = theta_samples.unsqueeze(0)

        if postprocessing is not None:
            B = theta_samples.shape[0]
            theta_samples = postprocessing(theta_samples.reshape(-1, D),
                                           config)
            theta_samples = theta_samples.view(B, config["nb_samples"], -1)

        for n in range(theta_samples.shape[0]):
            folder_plot = config['folderpath'] / 'posterior_distributions'
            folder_plot.mkdir(exist_ok=True, parents=True)

            if postprocessing is None:
                plot_posterior_distribution(
                    theta_samples[n].detach().cpu(),
                    config,
                    postprocessing=False,
                    ground_truth=None
                    if theta_gt is None else theta_gt[n].detach().cpu(),
                    idx=n,
                    fig_file=folder_plot / 'posterior_distribution')
            else:
                plot_posterior_distribution(
                    theta_samples[n].detach().cpu(),
                    config,
                    postprocessing=True,
                    ground_truth=None
                    if theta_gt is None else theta_gt[n].detach().cpu(),
                    idx=n,
                    fig_file=folder_plot /
                    'posterior_distribution_postprocessing')

    return map, mask, degeneracy_mask, uncertainty, ambiguity


def sample_posterior_distribution_rejection_sampling(
    x: torch.Tensor,
    config: Dict,
    nf: nn.Module,
    embedded_net: nn.Module,
    x_normalizer,
    theta_normalizer,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Batched rejection sampling from normalizing flow (used in old version).

    Parameters
    ----------
    x : torch.Tensor, shape (N, size_x)
        Observed diffusion MRI signal for N voxels. Its size must be identical to
        the signals used for training (i.e. ``config['size_x']``).

    config : dict
        μGUIDE configuration.

    nf : nn.Module
        Normalizing flow model.

    embedded_net : nn.Module
        Neural network that computes the embedding of the observed signal used as
        context for the normalizing flow.
    
    x_normalizer : Normalizer
        Normalizer for the observed signal x, used to compute the embedding.

    theta_normalizer : Normalizer
        Normalizer for the microstructure parameters theta, used to sample from
        the normalizing flow in the normalized space and then convert back to the
        original space.

    Returns
    -------
    samples : torch.Tensor, shape (B, S, D)
    """

    device = config['device']

    if x.ndim == 1:
        x = x.unsqueeze(0)  # Add batch dimension if only one voxel is provided

    x = x.to(device)

    B = x.shape[0]
    S = config["nb_samples"]
    D = config["size_theta"]

    if config['size_x'] != x.shape[-1]:
        raise ValueError('x size set in config does not match x size used ' \
                         'for training')

    # Normalize data and compute embedding
    with torch.no_grad():
        x_norm = x_normalizer.to(device)(x)
        embedding = embedded_net.to(device)(x_norm).detach()
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)

    theta_normalizer = theta_normalizer.to('cpu')

    # Get prior bounds for rejection sampling
    prior_min = torch.tensor(
        [config['prior'][p][0] for p in config['prior'].keys()], device=device)
    prior_max = torch.tensor(
        [config['prior'][p][1] for p in config['prior'].keys()], device=device)

    # Base distribution
    base_dist = dist.Normal(
        torch.zeros((B, D), device=device),
        torch.ones((B, D), device=device),
    ).to_event(1)
    transformed_dist = dist.ConditionalTransformedDistribution(base_dist, nf)
    with torch.inference_mode():
        cond_dist = transformed_dist.condition(embedding)

    # Storage
    chunk_size = config[
        "nb_samples"]  # Start with the full number of samples, will be reduced if OOM occurs
    z_buffer = torch.empty((chunk_size, B, D), device=device)
    theta_buffer = torch.empty((B, chunk_size, D), device=device)
    accepted_buffer = torch.empty((B, chunk_size),
                                  dtype=torch.bool,
                                  device=device)
    samples = torch.empty((B, S, D), device=device)
    counts = torch.zeros(B, dtype=torch.long, device=device)

    # Rejection sampling
    max_iter = 3000
    fallback_iter = 50
    loop_iter = 0

    # Rejection loop for the current voxel batch
    while (counts < S).any() and loop_iter < max_iter:

        try:
            # sample candidates (batched)
            with torch.inference_mode():
                z = cond_dist.sample((chunk_size, ))  # (S, B, D)
                z_buffer.copy_(z.cpu(), non_blocking=True)  # (S, B, D)
                del z
                theta_buffer[:] = theta_normalizer.inverse(
                    z_buffer.permute(1, 0, 2))  # (B, S, D)

            # Check bounds
            accepted_buffer[:] = ((theta_buffer > prior_min) &
                                  (theta_buffer < prior_max)).all(
                                      dim=2)  # (B, S)

            # Fallback to avoid infinite loop if no samples are accepted for
            # some voxels after many iterations
            if loop_iter == fallback_iter:
                stuck_voxels = counts <= 100
                if verbose:
                    print(
                        f"Fallback triggered for {stuck_voxels.sum().item()} "
                        f"voxels out of {B}. Accepting all samples for these voxels."
                    )

                if stuck_voxels.any():
                    # Accept everything for those voxels
                    accepted_buffer[stuck_voxels] = True

            # Fill samples per voxel
            for b in range(B):

                n_accept = accepted_buffer[b].sum().item()
                if n_accept == 0:
                    continue

                n_remaining = S - counts[b]
                n_take = min(n_accept, n_remaining)

                samples[b, counts[b]:counts[b] +
                        n_take] = theta_buffer[b][accepted_buffer[b]][:n_take]

                counts[b] += n_take

            loop_iter += 1

        except RuntimeError as e:
            if "out of memory" in str(e):
                chunk_size = max(1, chunk_size // 2)
                torch.cuda.empty_cache()
                print(f"[OOM] Reducing chunk_size → {chunk_size}")
                continue
            else:
                raise e

    if verbose:
        print(f"Rejection sampling completed in {loop_iter} iterations.")

    # Safety fallback
    if (counts < S).any():
        for b in range(B):
            if counts[b] < S:
                if verbose:
                    print(
                        f"Warning: rejection sampling incomplete for voxel {b},"
                        " filling with last samples")
                samples[b, counts[b]:] = samples[b, :counts[b]].mean(dim=0)

    return samples


def estimate_theta(
    samples: torch.Tensor,
    config: Dict,
    postprocessing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """
    Estimate MAP, mask, degeneracy, uncertainty and ambiguity
    from posterior samples (batched).

    Parameters
    ----------
    samples : torch.Tensor, shape (B, S, D)
        Posterior samples for B voxels in batch, with S samples each, and D microstructure parameters.

    config : dict
        μGUIDE configuration.

    postprocessing : bool, default=False
        Whether the samples have been postprocessed to convert surrogate parameters
        back to the original microstructure parameters. If True, the function will
        check the size of the samples against the size of the postprocessed prior
        instead of the original prior.

    Returns
    -------
    map_est : (B, D)
    mask : (B, D)
    degeneracy_mask : (B, D)
    uncertainty : (B, D)
    ambiguity : (B, D)
    """

    device = config['device']
    B, S, D = samples.shape

    prior = (config["prior_postprocessing"]
             if postprocessing else config["prior"])

    prior_min = torch.tensor([prior[p][0] for p in prior.keys()],
                             device=device)
    prior_max = torch.tensor([prior[p][1] for p in prior.keys()],
                             device=device)

    # Initialize map with mean values, will be updated for non-degenerate cases
    theta_mean = samples.mean(dim=1)  # (B, D)
    map_est = theta_mean.clone()

    # Mask for parameters where the mean is outside the prior bounds (estimation failure)
    mask = (theta_mean >= prior_min) & (theta_mean <= prior_max)

    # Uncertainty (IQR)
    q1 = torch.quantile(samples, 0.25, dim=1)
    q3 = torch.quantile(samples, 0.75, dim=1)
    uncertainty = (q3 - q1) / (prior_max - prior_min) * 100

    # Initialize degeneracy mask and ambiguity with default values (non-degenerate, 100% ambiguity)
    degeneracy_mask = torch.zeros((B, D), dtype=torch.bool, device=device)
    ambiguity = torch.ones((B, D), device=device) * 100

    # Gaussian fitting to determine degeneracy and update MAP and ambiguity
    # for non-degenerate cases
    for b in range(B):
        # Loop only over non-masked/valid parameters
        for d in torch.where(mask[b])[0]:

            # Compute histogram for the d-th parameter of the b-th voxel to
            # check for degeneracy
            hist, edges = _histogram_1d(samples[b, :, d],
                                        bins=80,
                                        density=True)
            hist_smooth = smooth_histogram(hist, kernel_size=7)
            x_hist = (edges[:-1] + edges[1:]) / 2  # Midpoints of bins

            # Fit two Gaussians to the histogram using scipy (CPU)
            param_gauss = fit_two_gaussians(
                x_hist.cpu().numpy(),
                hist_smooth.cpu().numpy(),
            )

            # Failed fit: set mask to False and skip degeneracy and ambiguity estimation
            if param_gauss[0] == 0 and param_gauss[3] == 0:
                mask[b, d] = False
                continue

            param_gauss_t = torch.tensor(param_gauss, device=device)

            bounds = prior[list(prior)[d]]

            degeneracy_mask[b, d] = is_degenerate(param_gauss_t, bounds,
                                                  config)
            map_est[b, d] = estimate_max_a_posteriori(param_gauss_t, bounds,
                                                      config)

            if not degeneracy_mask[b, d]:
                ambiguity[b, d] = estimate_ambiguity(param_gauss_t, bounds,
                                                     config)

    # Set uncertainty and ambiguity to 100% for degenerate cases and for masked/invalid cases
    uncertainty[~mask] = 100
    ambiguity[~mask] = 100
    uncertainty[degeneracy_mask] = 100
    ambiguity[degeneracy_mask] = 100

    return map_est, mask, degeneracy_mask, uncertainty, ambiguity


def is_degenerate(param_gauss, prior_bounds, config, num_points=1000):
    """
    Determines degeneracy of a Gaussian mixture by checking for multiple 
    significant peaks.

    Parameters
    ----------
    param_gauss (torch.Tensor):
        Tensor of Gaussian parameters. Shape (6,) corresponding to 
        (f1, mu1, sigma1, f2, mu2, sigma2).
    prior_bounds (torch.Tensor): 
        Tensor of min/max bounds, shape (num_params, 2).
    num_points (int):
        Number of points for evaluating the Gaussians.

    Returns
    --------
    torch.Tensor
        Boolean tensor indicating degeneracy for each parameter, shape (num_params,).
    """
    device = config['device']

    degenerate = torch.zeros((), dtype=torch.bool, device=device)

    x = torch.linspace(prior_bounds[0],
                       prior_bounds[1],
                       num_points,
                       device=device)
    der = derivative_two_gaussians(x, *param_gauss)

    sign_d = sign_der(der)
    idx_der = torch.nonzero(sign_d[:-1] != sign_d[1:],
                            as_tuple=False).flatten() + 1

    # If idx_der contain two consecutive numbers, means it is a suprious spike.
    # Do not take it into account
    if idx_der.numel() > 1:
        diffs = idx_der[1:] - idx_der[:-1]
        consecutive_with_prev = torch.cat(
            (torch.zeros(1, dtype=torch.bool, device=device), diffs == 1))
        consecutive_with_next = torch.cat(
            (diffs == 1, torch.zeros(1, dtype=torch.bool, device=device)))
        keep = ~(consecutive_with_prev | consecutive_with_next)
        idx_der = idx_der[keep]

    if idx_der.numel() > 1:
        # If distance between the mean of the two gaussians < sum of std,
        # then the two are too close to distinguish them. Not degenerate
        dist_mean = torch.abs(param_gauss[1] - param_gauss[4])
        if dist_mean > param_gauss[2] + param_gauss[5]:
            degenerate = torch.ones((), dtype=torch.bool, device=device)

    return degenerate


def estimate_max_a_posteriori(param_gauss,
                              prior_bounds,
                              config,
                              num_points=1000):
    """
    Estimates the maximum-a-posteriori (MAP) from the parameters of a Gaussian
    mixture by evaluating the mixture on a grid and finding the maximum.
    
    Parameters
    ----------
    param_gauss (torch.Tensor)
        Tensor of Gaussian parameters, shape (num_params, 6).
    prior_bounds (torch.Tensor)
    Tensor of min/max bounds, shape (num_params, 2).
    num_points (int)
        Number of points for evaluating the Gaussians.
    Returns
    --------
    torch.Tensor
        Estimated MAP value for each parameter, shape (num_params,).
    """
    x = torch.linspace(prior_bounds[0],
                       prior_bounds[1],
                       num_points,
                       device=config['device'])
    y = two_gaussians(x, *param_gauss)
    map = x[torch.argmax(y)]
    # check if multiple values in map (i.e. multiple peaks with same max value)
    if map.numel() > 1:
        print(f'Multiple peaks with same max value found. map = {map}')
        map = map[0]  # Take the first of the peaks' locations as MAP estimate
    return map


def estimate_ambiguity(param_gauss, prior_bounds, config, num_points=1000):
    """
    Computes ambiguity as the percentage of the domain where the posterior is above half-max.
    Vectorized with PyTorch.
    
    Parameters
    ----------
    param_gauss (torch.Tensor)
        Tensor of Gaussian parameters, shape (num_params, 6).
    prior_bounds (torch.Tensor)
        Tensor of min/max bounds, shape (num_params, 2).
    config (dict)
        Configuration dictionary.
    num_points (int)
        Number of points for evaluating the Gaussians.

    Returns
    --------
    torch.Tensor
        Ambiguity measure for each parameter, shape (num_params,).
    """
    # x = np.linspace(prior_bounds[0], prior_bounds[1], num_points)
    #
    # # ambiguity = (len(np.where(y > y.max()/2)[0]) / x.shape[0]) * 100
    # # return ambiguity
    # return (np.count_nonzero(y > y.max() / 2) / x.shape[0]) * 100

    x = torch.linspace(prior_bounds[0],
                       prior_bounds[1],
                       num_points,
                       device=config['device'])
    y = two_gaussians(x, *param_gauss)

    half_max = y.max(dim=0, keepdim=True)[0] / 2
    ambiguity = (torch.sum(y > half_max, dim=0).float() /
                 num_points) * 100  # Percentage of domain above half-max
    return ambiguity


def one_gaussian(x, f, mu, sigma):
    sigma = torch.clamp(sigma, min=1e-6)
    SQRT_2PI = torch.sqrt(
        torch.tensor(2.0 * torch.pi, dtype=torch.float32, device=x.device))
    return f * 1 / (sigma * SQRT_2PI) * torch.exp(-1 / 2 *
                                                  ((x - mu) / sigma)**2)


def two_gaussians(x, f1, mu1, sigma1, f2, mu2, sigma2):
    return one_gaussian(x, f1, mu1, sigma1) + one_gaussian(x, f2, mu2, sigma2)


def derivative_one_gaussian(x, f, mu, sigma):
    return -(x - mu) / sigma**2 * one_gaussian(x, f, mu, sigma)


def derivative_two_gaussians(x, f1, mu1, sigma1, f2, mu2, sigma2):
    return derivative_one_gaussian(
        x, f1, mu1, sigma1) + derivative_one_gaussian(x, f2, mu2, sigma2)


def sign_der(derivative):
    s = torch.sign(derivative)
    if s.numel() == 0:
        return s

    if s[0] == 0:
        nonzero = torch.nonzero(s != 0, as_tuple=False)
        if nonzero.numel() == 0:
            return s
        s[0] = s[nonzero[0, 0]]

    for i in range(1, s.shape[0]):
        if s[i] == 0:
            s[i] = s[i - 1]
    return s


def _histogram_1d(x: torch.Tensor,
                  bins: int = 80,
                  density: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA-safe 1D histogram.
    Uses torch.histc (CUDA-supported) and returns (hist, bin_edges) on x.device.
    """
    device = x.device
    x1 = x.detach().flatten().to(torch.float32)
    if x1.numel() == 0:
        hist = torch.zeros((bins, ), device=device, dtype=torch.float32)
        bin_edges = torch.linspace(0.0,
                                   1.0,
                                   bins + 1,
                                   device=device,
                                   dtype=torch.float32)
        return hist, bin_edges

    x_min = torch.min(x1)
    x_max = torch.max(x1)
    # histc requires min < max; if all values equal, widen slightly
    if torch.isclose(x_min, x_max):
        eps = torch.tensor(1e-6, device=device, dtype=torch.float32)
        x_min = x_min - eps
        x_max = x_max + eps

    try:
        # Fast path: histc (usually CUDA-supported)
        hist = torch.histc(x1, bins=bins, min=float(x_min), max=float(x_max))
        bin_edges = torch.linspace(float(x_min),
                                   float(x_max),
                                   bins + 1,
                                   device=device,
                                   dtype=torch.float32)

        if density:
            bin_width = (bin_edges[1] - bin_edges[0]).clamp_min(1e-12)
            hist_sum = hist.sum().clamp_min(1e-12)
            hist = hist / (hist_sum * bin_width)

        return hist, bin_edges

    except Exception as e:
        # Fallback: CPU histogram, then move results back to original device.
        x_cpu = x1.detach().to("cpu")
        hist_cpu, bin_edges_cpu = torch.histogram(x_cpu,
                                                  bins=bins,
                                                  density=density)
        hist = hist_cpu.to(device)
        bin_edges = bin_edges_cpu.to(device)
        return hist, bin_edges


def one_gaussian_np(x, f, mu, sigma):
    return f * 1 / (sigma *
                    (np.sqrt(2 * np.pi))) * np.exp(-1 / 2 *
                                                   ((x - mu) / sigma)**2)


def two_gaussians_np(x, f1, mu1, sigma1, f2, mu2, sigma2):
    return one_gaussian_np(x, f1, mu1, sigma1) + one_gaussian_np(
        x, f2, mu2, sigma2)


def fit_two_gaussians(x_hist, hist):
    min_hist = x_hist[0]
    max_hist = x_hist[-1]
    try:
        param_gauss, _ = optimize.curve_fit(
            two_gaussians_np,
            x_hist,
            hist,
            bounds=([0.0, min_hist, 0.0, 0.0, min_hist, 0.0],
                    [np.inf, max_hist, max_hist, np.inf, max_hist, max_hist]))
    except Exception:
        param_gauss = np.zeros(6)
        # print(traceback.format_exc())
    return param_gauss


def smooth_histogram(hist, kernel_size=7):
    kernel = torch.ones(1, 1, kernel_size, device=hist.device) / kernel_size
    hist = hist.view(1, 1, -1)
    hist_smooth = F.conv1d(hist, kernel, padding=kernel_size // 2)
    return hist_smooth.view(-1)
