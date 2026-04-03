import torch
from typing import Optional, Tuple


def preprocess_data(
    theta: torch.Tensor,
    x: torch.Tensor,
    bvals: Optional[torch.Tensor] = None,
    normalize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess simulation data by removing invalid samples and optionally
    normalizing diffusion signals with respect to b0.

    Parameters
    ----------
    theta : torch.Tensor, shape (N, D_theta)
        Microstructure parameters.

    x : torch.Tensor, shape (N, D_x)
        Observed diffusion MRI signals.

    bvals : torch.Tensor, shape (D_x,), optional
        b-values associated with the diffusion signals. Required if
        `normalize=True`.

    normalize : bool, default=False
        If True, normalize signals by their mean b0 value.

    Returns
    -------
    theta : torch.Tensor
        Filtered (and possibly reduced) parameter tensor.

    x : torch.Tensor
        Filtered (and optionally normalized) signal tensor.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent.
    """

    theta = torch.as_tensor(theta, dtype=torch.float32)
    x = torch.as_tensor(x, dtype=torch.float32)

    # -----------------------------
    # Check data size
    # -----------------------------
    if x.shape[0] != theta.shape[0]:
        raise ValueError("Number of samples in theta and x do not match.")

    if normalize:
        if bvals is None:
            raise ValueError("bvals must be provided when normalize=True.")

        bvals = torch.as_tensor(bvals, dtype=torch.float32)

        if x.shape[1] != bvals.shape[0]:
            raise ValueError("x size does not match number of b-values.")

    # -----------------------------
    # Remove NaN / inf samples
    # -----------------------------
    finite_mask = torch.isfinite(x).all(dim=1) & torch.isfinite(theta).all(
        dim=1)

    x = x[finite_mask]
    theta = theta[finite_mask]

    # -----------------------------
    # Normalize with respect to b0
    # -----------------------------
    if normalize:

        b0_mask = bvals <= 1e-5
        x0 = x[:, b0_mask].mean(dim=1, keepdim=True)

        valid_b0 = torch.isfinite(x0).all(dim=1) & (x0.squeeze(1) > 0)

        x = x[valid_b0]
        theta = theta[valid_b0]
        x0 = x0[valid_b0]

        x = x / x0

        # Second pass (normalization can introduce NaNs/infs)
        finite_mask = torch.isfinite(x).all(dim=1) & torch.isfinite(theta).all(
            dim=1)

        x = x[finite_mask]
        theta = theta[finite_mask]

    return theta, x


def postprocess_SM(
    samples: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """
    Convert latent parameters (u0, u1) into physically meaningful
    diffusivities (De_par, De_perp).

    Parameters
    ----------
    samples : torch.Tensor, shape (N, D)
        Posterior samples in latent space.

    config : dict
        Configuration dictionary containing:
            - 'prior': parameter ordering
            - 'prior_postprocessing': parameter bounds

    Returns
    -------
    torch.Tensor, shape (N, D)
        Postprocessed samples with updated diffusivity parameters.
    """

    samples = samples.to(config["device"])

    prior_keys = list(config["prior"].keys())

    idx_u0 = prior_keys.index("u0")
    idx_u1 = prior_keys.index("u1")

    u0 = torch.clamp(samples[:, idx_u0], 0.0, 1.0)
    u1 = torch.clamp(samples[:, idx_u1], 0.0, 1.0)

    De_par_min = config["prior_postprocessing"]["De_par"][0]
    De_par_max = config["prior_postprocessing"]["De_par"][1]
    De_perp_min = config["prior_postprocessing"]["De_perp"][0]

    De_par = torch.sqrt((De_par_max - De_par_min)**2 * u0) + De_par_min
    De_perp = (De_par - De_par_min) * u1 + De_perp_min

    out_samples = samples.clone()

    out_samples[:, idx_u0] = De_par
    out_samples[:, idx_u1] = De_perp

    return out_samples


def postprocess_SANDI(
    samples: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """
    Convert SANDI latent parameters (k1, k2) into microstructural fractions
    (f_neurite, f_soma, f_extra).

    Parameters
    ----------
    samples : torch.Tensor, shape (N, D)
        Posterior samples containing k1 and k2 parameters.

    config : dict
        Configuration dictionary containing:
            - 'prior': parameter ordering
            - 'device': computation device

    Returns
    -------
    torch.Tensor, shape (N, D+1)
        Postprocessed samples with:
            - f_neurite (fn)
            - f_soma (fs)
            - f_extra (fe)
            appended in the first dimensions.
    """

    samples = samples.to(config["device"])

    prior_keys = list(config["prior"].keys())

    idx_k1 = prior_keys.index("k1")
    idx_k2 = prior_keys.index("k2")

    k1 = torch.clamp(samples[:, idx_k1], min=0.0)
    k2 = torch.clamp(samples[:, idx_k2], min=0.0)

    sqrt_k1 = torch.sqrt(k1)

    fn = k2 * sqrt_k1
    fs = (1.0 - k2) * sqrt_k1
    fe = 1.0 - sqrt_k1

    samples_f = torch.zeros(
        (samples.shape[0], samples.shape[1] + 1),
        dtype=samples.dtype,
        device=config["device"],
    )

    samples_f[:, 0] = fn
    samples_f[:, 1] = fs
    samples_f[:, 2] = fe

    samples_f[:, 3:] = samples[:, 2:]

    return samples_f
