import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pyro.distributions as dist
import matplotlib.pyplot as plt

from uGUIDE.dataset import split_data
from uGUIDE.density_estimator import get_nf
from uGUIDE.embedded_net import get_embedded_net
from uGUIDE.normalization import get_normalizer, save_normalizer

def run_inference(theta, x, config, plot_loss=True, load_state=False):
    
    # check if size(x) is compatible with size within config file
    if config['size_theta'] != theta.shape[1]:
        raise ValueError('Theta size set in config is different from the ' \
                         'size of theta used for training')
    if config['size_x'] != x.shape[1]:
        raise ValueError('x size set in config is different from the size ' \
                         'of x used for training')

    # Normalize the data
    theta_normalizer = get_normalizer(theta)
    save_normalizer(theta_normalizer, config['folder_path'] / config['theta_normalizer_file'])
    x_normalizer = get_normalizer(x)
    save_normalizer(x_normalizer, config['folder_path'] / config['x_normalizer_file'])

    theta_norm = theta_normalizer(theta)
    x_norm = x_normalizer(x)
    theta_norm = torch.from_numpy(theta_norm).to(config['device'])
    x_norm = torch.from_numpy(x_norm).to(config['device'])
    
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
    embedded_net = get_embedded_net(input_dim=config['size_x'],
                                    output_dim=config['nf_features'],
                                    folder_path=config['folder_path'],
                                    embedder_state_dict_file=config['embedder_state_dict_file'],
                                    layer_1_dim=config['hidden_layers'][0],
                                    layer_2_dim=config['hidden_layers'][1],
                                    load_state=load_state)

    base_dist = dist.Normal(
        loc=torch.zeros(config['size_theta']), scale=torch.ones(config['size_theta'])
    )
    transformed_dist = dist.ConditionalTransformedDistribution(base_dist, nf)

    modules = torch.nn.ModuleList([nf, embedded_net])
    optimizer = torch.optim.Adam(modules.parameters(), lr=config['learning_rate'])

    best_val_loss = np.inf
    val_losses = []


    for _ in tqdm(range(config['epochs'])):

        modules.train()
        for theta_batch, x_batch in train_dataloader:
            optimizer.zero_grad()
            embedding = embedded_net(x_batch.detach().type(torch.float32))
            lp_theta = transformed_dist.condition(embedding).log_prob(
                theta_batch.detach().type(torch.float32)
            )
            loss = -lp_theta.mean()
            loss.backward()

            optimizer.step()
            transformed_dist.clear_cache()

        modules.eval()
        with torch.no_grad():
            loss_acc = []
            for theta_batch, x_batch in val_dataloader:
                embedding = embedded_net(x_batch.detach().type(torch.float32))
                lp_theta = transformed_dist.condition(embedding).log_prob(
                    theta_batch.detach().type(torch.float32)
                )
                loss = -lp_theta.mean()
                loss_acc.append(loss.detach().numpy())

            new_val_loss = np.mean(loss_acc)
            if new_val_loss < best_val_loss:
                print("New best val loss: ", new_val_loss)
                best_val_loss = new_val_loss
                torch.save(
                    embedded_net.state_dict(),
                    config['folder_path'] / config['embedder_state_dict_file']
                )
                torch.save(
                    nf.state_dict(), 
                    config['folder_path'] / config['nf_state_dict_file']
                )

            val_losses.append(new_val_loss)

    if plot_loss:
        plt.plot(val_losses, label="- Forward KL")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.yscale("symlog")
        plt.legend()
        plt.show()
        plt.savefig

    return