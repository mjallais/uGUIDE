from torch import nn


def build_embedder_MLP(input_dim, output_dim, layer_1_dim=128, layer_2_dim=64):

    # Values from Louis' code 
    embedder = nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=layer_1_dim),
        nn.ReLU(),
        nn.Linear(in_features=layer_1_dim, out_features=layer_2_dim),
        nn.ReLU(),
        nn.Linear(in_features=layer_2_dim, out_features=output_dim),
    )

    return embedder