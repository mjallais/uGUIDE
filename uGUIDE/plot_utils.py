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
    for p, param in enumerate(prior.keys()):

        hist_sbi, bin_edges = np.histogram(samples[:,p], density=False, bins=50)
        n = len(hist_sbi)
        x_hist_sbi=np.zeros((n),dtype=float)
        for ii in range(n):
            x_hist_sbi[ii]=(bin_edges[ii+1]+bin_edges[ii])/2
        axs[p].plot(x_hist_sbi, hist_sbi)
        axs[p].fill_between(x_hist_sbi, hist_sbi, alpha=0.4)
        if ground_truth is not None:
            axs[p].axvline(ground_truth[p], linestyle='--', color='k')
        axs[p].set_xlabel(param, fontsize=20)
        axs[p].set_xlim(prior[param][0], prior[param][1])
        axs[p].set_yticks([])
        axs[p].tick_params(axis='x', which='major', labelsize=20)

    fig.tight_layout()
    plt.savefig(config['folderpath'] / fig_file)
    
    return
