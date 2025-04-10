import numpy as np
import torch

def preprocess_data(theta, x, bvals, normalize=False):

    # Check data size
    if x.shape[0] != theta.shape[0]:
        raise ValueError('Number of samples in theta and x do not match.')
    if normalize == True:
        if x.shape[1] != bvals.shape[0]:
            raise ValueError('x size does not match the number of b-values.')

    # Remove nan and inf present in the input signals
    for data in [x, theta]:
        idx_nan = np.where(np.isnan(data))
        x = np.delete(x, idx_nan[0], 0)
        theta = np.delete(theta, idx_nan[0], 0)

        idx_inf = np.where(np.isinf(data))
        x = np.delete(x, idx_inf[0], 0)
        theta = np.delete(theta, idx_inf[0], 0)

    # Normalize signal wrt b0
    if normalize == True:
        x0 = x[:, bvals == 0].mean(1, keepdims=True)
        x_norm = x / x0
        x = x_norm

    return theta, x


def postprocess_SM(samples, config):

    # Convert u0 and u1 into De_par and De_perp
    prior_keys = list(config['prior'].keys())
    u0 = samples[:, prior_keys.index('u0')]
    u1 = samples[:, prior_keys.index('u1')]
    # Set negative values to 0, otherwise get nan values
    u0 = torch.clip(u0, 0, 1)
    u1 = torch.clip(u1, 0, 1)
    De_par_min = config['prior_postprocessing']['De_par'][0]
    De_par_max = config['prior_postprocessing']['De_par'][1]
    De_perp_min = config['prior_postprocessing']['De_perp'][0]
    De_par = torch.sqrt((De_par_max - De_par_min)**2 * u0) + De_par_min
    De_perp = (De_par - De_par_min) * u1 + De_perp_min
    
    out_samples = samples.detach().clone()
    out_samples[:,prior_keys.index('u0')] = De_par
    out_samples[:,prior_keys.index('u1')] = De_perp

    return out_samples.to(config['device'])


def postprocess_SANDI(samples, config):

    # Convert k1 and k2 into fn, fs and fe
    prior_keys = list(config['prior'].keys())
    k1 = samples[:, prior_keys.index('k1')]
    k2 = samples[:, prior_keys.index('k2')]
    # Set negative values to 0, otherwise get nan values
    k1[k1<0] = 0
    k2[k2<0] = 0
    fn = k2 * torch.sqrt(k1)
    fs = (1 - k2) * torch.sqrt(k1)
    fe = 1 - torch.sqrt(k1)

    samples_f = torch.zeros((samples.shape[0], samples.shape[1]+1),
                            dtype=samples.dtype, device=config['device'])
    samples_f[:,0] = fn[:,0]
    samples_f[:,1] = fs[:,0]
    samples_f[:,2] = fe[:,0]
    samples_f[:,3:] = samples[:,2:]

    return samples_f
