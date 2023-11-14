import numpy as np
import torch
from torch.utils.data import DataLoader

from uGUIDE.dataset import split_data
from uGUIDE.density_estimator import build_nf
from uGUIDE.embedded_net import build_embedder_MLP

def run_inference(theta, x, device='cpu'):

    theta = torch.Tensor(theta).to(device)
    x = torch.Tensor(x).to(device)

    # Normalize the data

    train_dataset, val_dataset = split_data(theta, X)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4_096)

    nf = build_nf()
    # embedding_net = nn.Identity()
    embedding_net = build_embedder_MLP()

    # line 70 of Louis' code run_sbi

    return