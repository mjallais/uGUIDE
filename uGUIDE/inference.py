import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt

from uGUIDE.dataset import split_data
from uGUIDE.density_estimator import get_nf
from uGUIDE.embedded_net import get_embedded_net
from uGUIDE.normalization import get_normalizer, save_normalizer


def run_inference(theta, x, config, plot_loss=True, load_state=False):
    """
    Run inference given a training dataset.

    Parameters
    ----------
    theta : ndarray, shape (nb_simulations, config['size_theta'])
        Microstructure parameters used for the training. 
    
    x : ndarray, shape (nb_simulations, config['size_x'])
        Observed diffusion MRI signals corresponding to the theta.

    config : dict
        μGUIDE configuration.

    plot_loss : bool, default=True
        Plot the validation loss evolution during training.

    load_state: bool, default=False
        If training has already been performed, load the state
        of the neural networks saved if set to ``True``, and continue training.
        Otherwise, start a new inference.

    """

    # check if size(x) is compatible with size within config file
    if config['size_theta'] != theta.shape[1]:
        raise ValueError('Theta size set in config does not match theta size ' \
                         'used for training')
    if config['size_x'] != x.shape[1]:
        raise ValueError('x size set in config does not match x size used ' \
                         'for training')

    pyro.set_rng_seed(config['random_seed'])

    # Normalize the data
    theta_normalizer = get_normalizer(theta)
    save_normalizer(theta_normalizer,
                    config['folderpath'] / config['theta_normalizer_file'])
    x_normalizer = get_normalizer(x)
    save_normalizer(x_normalizer,
                    config['folderpath'] / config['x_normalizer_file'])

    theta_norm = theta_normalizer(theta)
    x_norm = x_normalizer(x)

    # Split training/validation sets
    train_dataset, val_dataset = split_data(theta_norm, x_norm)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4_096)
    # Initialize NF and the embedded neural network
    # Or load pretrained ones if load_state=True
    if load_state == True:
        nf = get_nf(input_dim=config['size_theta'],
                    nf_features=config['nf_features'],
                    pretrained_state=config['folderpath'] /
                    config['nf_state_dict_file'])
        embedded_net = get_embedded_net(input_dim=config['size_x'],
                                        output_dim=config['nf_features'],
                                        layer_1_dim=config['hidden_layers'][0],
                                        layer_2_dim=config['hidden_layers'][1],
                                        pretrained_state=config['folderpath'] /
                                        config['embedder_state_dict_file'],
                                        use_MLP=config['use_MLP'])
    else:
        nf = get_nf(input_dim=config['size_theta'],
                    nf_features=config['nf_features'],
                    pretrained_state=None)
        embedded_net = get_embedded_net(input_dim=config['size_x'],
                                        output_dim=config['nf_features'],
                                        layer_1_dim=config['hidden_layers'][0],
                                        layer_2_dim=config['hidden_layers'][1],
                                        pretrained_state=None,
                                        use_MLP=config['use_MLP'])
    nf.to(config['device'])
    embedded_net.to(config['device'])

    base_dist = dist.Normal(
        loc=torch.zeros(config['size_theta']).to(config['device']),
        scale=torch.ones(config['size_theta']).to(config['device']))
    transformed_dist = dist.ConditionalTransformedDistribution(base_dist, nf)

    modules = torch.nn.ModuleList([nf, embedded_net])
    optimizer = torch.optim.Adam(modules.parameters(),
                                 lr=config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['scheduler_patience'],
        min_lr=1e-5,
        verbose=True,
    )

    best_val_loss = np.inf
    val_losses = []
    lr_history = []
    epoch = 0
    epochs_no_change = 0

    pbar = tqdm(desc='Run inference', total=config['max_epochs'])
    while epoch < config['max_epochs'] \
        and epochs_no_change < config['n_epochs_no_change']:

        modules.train()
        for theta_batch, x_batch in train_dataloader:
            if torch.isnan(theta_batch).any() or torch.isinf(
                    theta_batch).any():
                print('NaN or inf values found in theta batch')
                break
            if torch.isnan(x_batch).any() or torch.isinf(x_batch).any():
                print('NaN or inf values found in x batch')
                break

            optimizer.zero_grad()
            embedding = embedded_net(x_batch.detach().type(torch.float32).to(
                config['device']))
            embedding = torch.tanh(embedding)
            lp_theta = transformed_dist.condition(embedding).log_prob(
                theta_batch.detach().type(torch.float32).to(config['device']))
            loss = -lp_theta.mean()

            if not torch.isfinite(loss):
                print("Skipping batch (NaN loss)")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(modules.parameters(), max_norm=1.0)

            optimizer.step()
            transformed_dist.clear_cache()

        modules.eval()
        with torch.no_grad():
            loss_acc = []
            for theta_batch, x_batch in val_dataloader:
                embedding = embedded_net(x_batch.detach().type(
                    torch.float32).to(config['device']))
                lp_theta = transformed_dist.condition(embedding).log_prob(
                    theta_batch.detach().type(torch.float32).to(
                        config['device']))
                loss = -lp_theta.mean()
                loss_acc.append(loss.item())

            new_val_loss = float(np.mean(loss_acc))

            # Step the scheduler and check if learning rate was reduced
            old_lr = optimizer.param_groups[0]['lr']
            lr_history.append(old_lr)
            scheduler.step(new_val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            # Reset counter if learning rate was reduced
            if new_lr < old_lr:
                print(f"LR reduced: {old_lr:.2e} → {new_lr:.2e}")
                epochs_no_change = 0

            # Early stopping and save best model
            if new_val_loss < best_val_loss:
                best_val_loss = new_val_loss
                epochs_no_change = 0
                torch.save(
                    embedded_net.state_dict(),
                    config['folderpath'] / config['embedder_state_dict_file'])
                torch.save(nf.state_dict(),
                           config['folderpath'] / config['nf_state_dict_file'])
            else:
                epochs_no_change += 1

            val_losses.append(new_val_loss)
        epoch += 1
        pbar.set_postfix_str(f'Best val loss = {best_val_loss}')
        pbar.update()
    pbar.close()
    print(f'Inference done. Convergence reached after {epoch} epochs.')

    if plot_loss:
        fig, ax1 = plt.subplots()

        # Loss (left axis)
        ax1.plot(val_losses, label="Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_yscale("symlog")

        # LR (right axis)
        ax2 = ax1.twinx()
        ax2.plot(lr_history, linestyle='--', label="Learning Rate")
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale("log")

        # Legend
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)

        plt.title("LR vs Loss during training")

        fig.savefig(config['folderpath'] / 'loss_training.png')

    return
