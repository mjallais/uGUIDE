import torch
from torch import nn

def build_embedder_MLP(input_dim, output_dim, layer_1_dim=128, layer_2_dim=64):

    embedder = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=layer_1_dim),
        nn.ReLU(),
        nn.Linear(in_features=layer_1_dim, out_features=layer_2_dim),
        nn.ReLU(),
        nn.Linear(in_features=layer_2_dim, out_features=output_dim),
    )

    return embedder

def get_embedded_net(input_dim, output_dim, layer_1_dim=128, layer_2_dim=64,
                     pretrained_state=None, use_MLP=True):

    if use_MLP == False:
        embedded_net = nn.Identity()
    else:
        embedded_net = build_embedder_MLP(input_dim=input_dim, output_dim=output_dim, 
                                        layer_1_dim=layer_1_dim, layer_2_dim=layer_2_dim)
    
        if pretrained_state is not None:
            embedder_state_dict = torch.load(pretrained_state)
            embedded_net.load_state_dict(embedder_state_dict)
            embedded_net.eval()
        
    return embedded_net
