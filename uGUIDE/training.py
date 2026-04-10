import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt

from uGUIDE.dataset import split_data
from uGUIDE.density_estimator import get_nf
from uGUIDE.embedded_net import get_embedded_net
from uGUIDE.normalization import get_normalizer, save_normalizer, get_bounded_normalizer, save_bounded_normalizer


def run_training(theta, x, config, plot_loss=True, load_state=False):
    """
    Run training given a training dataset.

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
    torch.set_num_threads(config['num_threads'])

    device = config['device']

    # Data (CPU only, then GPU tensors for training in batches if GPU available)
    theta = theta.cpu()
    x = x.cpu()

    # Normalize the data
    theta_normalizer = get_bounded_normalizer(config)
    save_bounded_normalizer(
        theta_normalizer,
        config['folderpath'] / config['theta_normalizer_file'])
    x_normalizer = get_normalizer(x, use_log1p=True, clip_value=5.0)
    save_normalizer(x_normalizer,
                    config['folderpath'] / config['x_normalizer_file'])

    theta_norm = theta_normalizer(theta)
    x_norm = x_normalizer(x)

    # Split training/validation sets
    num_workers = 4 if config['device'] == 'cuda' else 0
    train_dataset, val_dataset = split_data(theta_norm, x_norm)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1_024,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  persistent_workers=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=4_096,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=True)

    # Models
    nf = get_nf(input_dim=config['size_theta'],
                nf_features=config['nf_features'],
                n_flows=config['n_flows'],
                pretrained_state=config['folderpath'] /
                config['nf_state_dict_file'] if load_state else None)
    embedded_net = get_embedded_net(
        input_dim=config['size_x'],
        output_dim=config['nf_features'],
        layer_1_dim=config['hidden_layers'][0],
        layer_2_dim=config['hidden_layers'][1],
        pretrained_state=config['folderpath'] /
        config['embedder_state_dict_file'] if load_state else None,
        use_MLP=config['use_MLP'])

    nf.to(device)
    embedded_net.to(device)

    base_dist = dist.Normal(loc=torch.zeros(config['size_theta']).to(device),
                            scale=torch.ones(
                                config['size_theta']).to(device)).to_event(1)
    transformed_dist = dist.ConditionalTransformedDistribution(base_dist, nf)

    modules = torch.nn.ModuleList([nf, embedded_net])
    optimizer = torch.optim.Adam(modules.parameters(),
                                 lr=config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['scheduler_patience'],
        min_lr=config['learning_rate_min'],
        verbose=True,
    )

    # Training
    best_val_loss = float('inf')
    val_losses = []
    lr_history = []
    epochs_no_change = 0
    epoch = 0

    pbar = tqdm(desc='Training', total=config['max_epochs'])
    while epoch < config['max_epochs'] \
        and epochs_no_change < config['n_epochs_no_change']:

        modules.train()

        for theta_batch, x_batch in train_dataloader:

            theta_batch = theta_batch.to(device, non_blocking=True)
            x_batch = x_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            embedding = embedded_net(x_batch)
            embedding = torch.tanh(
                embedding)  # Ensure embedding is bounded between -1 and 1
            embedding = F.layer_norm(embedding, embedding.shape[-1:])

            cond_dist = transformed_dist.condition(embedding)

            with pyro.validation_enabled(False):
                lp_theta = cond_dist.log_prob(theta_batch)

            invalid_ratio = (~torch.isfinite(lp_theta)).float().mean()
            if invalid_ratio > 0:
                print(f"{invalid_ratio*100:.2f}% invalid log_probs")

            lp_theta = lp_theta[torch.isfinite(lp_theta)]
            if len(lp_theta) == 0:
                continue

            loss = -lp_theta.mean()

            if not torch.isfinite(loss):
                print("Skipping batch (NaN loss)")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(modules.parameters(), max_norm=1.0)

            optimizer.step()

        # Validation
        modules.eval()
        loss_acc = []

        with torch.no_grad():
            for theta_batch, x_batch in val_dataloader:

                theta_batch = theta_batch.to(device, non_blocking=True)
                x_batch = x_batch.to(device, non_blocking=True)

                embedding = embedded_net(x_batch)
                embedding = torch.tanh(embedding)
                embedding = F.layer_norm(embedding, embedding.shape[-1:])

                cond_dist = transformed_dist.condition(embedding)

                with pyro.validation_enabled(False):
                    lp_theta = cond_dist.log_prob(theta_batch)

                lp_theta = lp_theta[torch.isfinite(lp_theta)]
                if len(lp_theta) == 0:
                    continue

                loss = -lp_theta.mean()
                loss_acc.append(loss.item())

            new_val_loss = float(np.mean(loss_acc))
            val_losses.append(new_val_loss)

            # Step the scheduler and check if learning rate was reduced
            old_lr = optimizer.param_groups[0]['lr']
            lr_history.append(old_lr)
            scheduler.step(new_val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            # Reset counter if learning rate was reduced
            if new_lr < old_lr:
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

        epoch += 1
        pbar.set_postfix_str(f'Best val loss = {best_val_loss}')
        pbar.update()

    pbar.close()
    del train_dataloader
    del val_dataloader
    torch.cuda.empty_cache()

    print(f'Training done. Convergence reached after {epoch} epochs.')

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

    return best_val_loss
