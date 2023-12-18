import torch
from torch import nn


# called "embedded" in filename

def build_embedder_MLP(input_dim, output_dim, layer_1_dim=128, layer_2_dim=64):

    # Need to ensure input_dim > layer1 > layer2 > output_dim
    # why ?

    # Values from Louis' code
    # perhaps better to have some variable config, like a hidden_dims list ? as you wish
    embedder = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=layer_1_dim),
        nn.ReLU(),
        nn.Linear(in_features=layer_1_dim, out_features=layer_2_dim),
        nn.ReLU(),
        nn.Linear(in_features=layer_2_dim, out_features=output_dim),
    )

    return embedder

def get_embedded_net(input_dim, output_dim,
                     layer_1_dim=128, layer_2_dim=64, pretrained_state=False):

    embedded_net = build_embedder_MLP(input_dim=input_dim, output_dim=output_dim, 
                                       layer_1_dim=128, layer_2_dim=64)
    
    if pretrained_state is not None:
        embedder_state_dict = torch.load(pretrained_state)
        embedded_net.load_state_dict(embedder_state_dict)
        embedded_net.eval()
    
    return embedded_net
