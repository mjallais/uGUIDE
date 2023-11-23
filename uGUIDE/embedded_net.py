import torch
from torch import nn

def build_embedder_MLP(input_dim, output_dim, layer_1_dim=128, layer_2_dim=64):

    # Need to ensure input_dim > layer1 > layer2 > output_dim

    # Values from Louis' code 
    embedder = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=layer_1_dim),
        nn.ReLU(),
        nn.Linear(in_features=layer_1_dim, out_features=layer_2_dim),
        nn.ReLU(),
        nn.Linear(in_features=layer_2_dim, out_features=output_dim),
    )

    return embedder

def get_embedded_net(input_dim, output_dim, folder_path, embedder_state_dict_file,
                     layer_1_dim=128, layer_2_dim=64, load_state=False):

    embedded_net = build_embedder_MLP(input_dim=input_dim, output_dim=output_dim, 
                                       layer_1_dim=128, layer_2_dim=64)
    
    if load_state:
        embedder_state_dict = torch.load(folder_path / embedder_state_dict_file)
        embedded_net.load_state_dict(embedder_state_dict)
        embedded_net.eval()
    
    return embedded_net
