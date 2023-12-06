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
    
    # check if size(x) is compatible with size within config file
    if config['size_theta'] != theta.shape[1]:
        raise ValueError('Theta size set in config dos not match theta size ' \
                         'used for training')
    if config['size_x'] != x.shape[1]:
        raise ValueError('x size set in config does not match x size used ' \
                         'for training')

    pyro.set_rng_seed(config['random_seed'])

    # Normalize the data
    theta_normalizer = get_normalizer(theta)
    save_normalizer(theta_normalizer, config['folder_path'] / config['theta_normalizer_file'])
    x_normalizer = get_normalizer(x)
    save_normalizer(x_normalizer, config['folder_path'] / config['x_normalizer_file'])

    theta_norm = theta_normalizer(theta)
    x_norm = x_normalizer(x)
    
    # Split training/validation sets
    train_dataset, val_dataset = split_data(theta_norm, x_norm)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4_096)
    # Initialize NF and the embedded neural network
    # Or load pretrained ones if load_state=True
    nf = get_nf(input_dim=config['size_theta'],
                nf_features=config['nf_features'],
                folder_path=config['folder_path'],
                nf_state_dict_file=config['nf_state_dict_file'],
                load_state=load_state)
    nf.to(config['device'])
    embedded_net = get_embedded_net(input_dim=config['size_x'],
                                    output_dim=config['nf_features'],
                                    folder_path=config['folder_path'],
                                    embedder_state_dict_file=config['embedder_state_dict_file'],
                                    layer_1_dim=config['hidden_layers'][0],
                                    layer_2_dim=config['hidden_layers'][1],
                                    load_state=load_state)
    embedded_net.to(config['device'])

    base_dist = dist.Normal(
        loc=torch.zeros(config['size_theta']).to(config['device']),
        scale=torch.ones(config['size_theta']).to(config['device'])
    )
    transformed_dist = dist.ConditionalTransformedDistribution(base_dist, nf)

    modules = torch.nn.ModuleList([nf, embedded_net])
    optimizer = torch.optim.Adam(modules.parameters(), lr=config['learning_rate'])

    best_val_loss = np.inf
    val_losses = []
    epoch = 0
    epochs_no_change = 0

    pbar = tqdm(desc='Run inference', total = config['max_epochs'])
    while epoch < config['max_epochs'] \
        and epochs_no_change < config['n_epochs_no_change']:
        
        modules.train()
        for theta_batch, x_batch in train_dataloader:
            optimizer.zero_grad()
            embedding = embedded_net(x_batch.detach().type(torch.float32).to(config['device']))
            lp_theta = transformed_dist.condition(embedding).log_prob(
                theta_batch.detach().type(torch.float32).to(config['device'])
            )
            loss = -lp_theta.mean()
            loss.backward()

            optimizer.step()
            transformed_dist.clear_cache()

        modules.eval()
        with torch.no_grad():
            loss_acc = []
            for theta_batch, x_batch in val_dataloader:
                embedding = embedded_net(x_batch.detach().type(torch.float32).to(config['device']))
                lp_theta = transformed_dist.condition(embedding).log_prob(
                    theta_batch.detach().type(torch.float32).to(config['device'])
                )
                loss = -lp_theta.mean()
                loss_acc.append(loss.detach().cpu().numpy())

            new_val_loss = np.mean(loss_acc)
            if new_val_loss < best_val_loss:
                best_val_loss = new_val_loss
                epochs_no_change = 0
                torch.save(
                    embedded_net.state_dict(),
                    config['folder_path'] / config['embedder_state_dict_file']
                )
                torch.save(
                    nf.state_dict(), 
                    config['folder_path'] / config['nf_state_dict_file']
                )
            else:
                epochs_no_change += 1

            val_losses.append(new_val_loss)
        epoch += 1
        pbar.set_postfix_str(f'Best val loss = {best_val_loss}')
        pbar.update()
    pbar.close()
    print(f'Inference done. Convergence reached after {epoch} epochs.')

    if plot_loss:
        plt.plot(val_losses, label="- Forward KL")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.yscale("symlog")
        plt.legend()
        plt.show()
        plt.savefig

    return
