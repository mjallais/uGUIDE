import numpy as np

def preprocess_data(theta, x, bvals):

    # Check data size
    if x.shape[0] != theta.shape[0]:
        raise ValueError('Number of samples in theta and x do not match.')
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
    x_norm=np.zeros_like(x)
    x0 = x[:, bvals == 0].mean(1)
    for i in np.arange(x.shape[0]):
        x_norm[i,:] = x[i,:] / x0[i]

    return theta, x_norm


def postprocess_SM(samples, config):

    idx_u0 = np.where(np.array(list(config['prior'].keys())) == 'u0')[0]
    idx_u1 = np.where(np.array(list(config['prior'].keys())) == 'u1')[0]
    u0 = samples[:,idx_u0]
    u1 = samples[:,idx_u1]
    # Set negative values to 0, otherwise get nan values
    u0 = np.where(u0 < 0, 0, u0)
    u1 = np.where(u1 < 0, 0, u1)
    De_par_min = config['prior_postprocessing']['De_par'][0]
    De_par_max = config['prior_postprocessing']['De_par'][1]
    De_perp_min = config['prior_postprocessing']['De_perp'][0]
    De_par = np.sqrt((De_par_max - De_par_min)**2 * u0) + De_par_min
    De_perp = (De_par - De_par_min)**2 * u1 + De_perp_min
    samples[:,idx_u0] = De_par
    samples[:,idx_u1] = De_perp

    return samples
