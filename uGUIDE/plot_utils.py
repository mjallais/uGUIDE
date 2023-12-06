import matplotlib.pyplot as plt
import seaborn as sns


def plot_posterior_distribution(samples, config,
                                fig_file='posterior_distribution.png'):

    # Check if samples have the save size as size_theta in config
    if config['size_theta'] != samples.shape[1]:
        raise ValueError('Theta size set in config does not match theta ' \
                         'size used for training')

    fig, axs = plt.subplots(
        nrows=1,
        ncols=config['size_theta'],
        figsize=(5 * config['size_theta'], 5),
        sharey="row"
    )

    for i, param in enumerate(config['prior'].keys()):
        sns.kdeplot(
            samples[:,i],
            alpha=0.5,
            fill=True,
            ax=axs[i],
            clip=config['prior'][param]
        )
        axs[i].set_xlabel(param, fontsize=20)
        axs[i].set_xlim(config['prior'][param][0], config['prior'][param][1])

    fig.tight_layout()
    plt.savefig(config['folder_path'] / fig_file)
    
    return

def plot_marginal_posterior(samples, config, ground_truth=None,
                            fig_file='marginal_posterior_distribution.png'):
    # Not working ; WIP
    
    if ground_truth is not None and ground_truth.shape != config['size_theta']:
        raise ValueError('Ground truth size does not match theta size set in '\
                         'config')

    for i, param in enumerate(config['prior'].keys()):
        plt.hist(samples[:, i])
        if ground_truth is not None:
            plt.axvline(ground_truth[i], '--', color='r')
    plt.show()

    return