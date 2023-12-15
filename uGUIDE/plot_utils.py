import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_posterior_distribution(samples, config,
                                postprocessing=True,
                                ground_truth=None,
                                fig_file='posterior_distribution.png'):

    # Check if samples have the save size as size_theta in config
    if config['size_theta'] != samples.shape[1]:
        # Maybe preprocessing step was applied that updated the size of the samples
        if (postprocessing == True) & (len(config['prior_postprocessing']) != samples.shape[1]):
            raise ValueError('Theta size set in config does not match theta ' \
                            'size used for training')
    
    if (ground_truth is not None) and (ground_truth.shape[0] != config['size_theta']):
        if (postprocessing == True) & (len(config['prior_postprocessing']) != samples.shape[1]):
            raise ValueError('Ground truth size does not match theta size set in '\
                            'config')

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
    for i, param in enumerate(prior.keys()):
        density = stats.gaussian_kde(samples[:,i], 'scott')
        xd = np.linspace(prior[param][0], prior[param][1], 100)
        yd = density(xd)
        axs[i].plot(xd, yd)
        axs[i].fill_between(xd, yd, alpha=0.5)
        if ground_truth is not None:
            axs[i].axvline(ground_truth[i], linestyle='--', color='k')
        axs[i].set_xlabel(param, fontsize=20)
        axs[i].set_xlim(prior[param][0], prior[param][1])

    fig.tight_layout()
    plt.savefig(config['folderpath'] / fig_file)
    
    return
