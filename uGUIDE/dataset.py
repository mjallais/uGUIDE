import torch
from torch.utils.data import Dataset


class uGUIDEDataset(Dataset):
    def __init__(self, theta, X) -> None:
        super().__init__()

        self.theta = torch.as_tensor(theta, dtype=torch.float32)
        self.X = torch.as_tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, index):
        return self.theta[index], self.X[index]
    

def split_data(theta, X, val_ratio=0.05):
    theta_t = torch.as_tensor(theta, dtype=torch.float32)
    X_t = torch.as_tensor(X, dtype=torch.float32)

    n_data = len(theta_t)
    n_data_val = int(n_data * val_ratio)

    perm = torch.randperm(n_data)

    theta_train = theta_t[perm][n_data_val:]
    theta_val = theta_t[perm][:n_data_val]

    X_train = X_t[perm][n_data_val:]
    X_val = X_t[perm][:n_data_val]

    train_dataset = uGUIDEDataset(
        theta_train,
        X_train
    )
    val_dataset = uGUIDEDataset(
        theta_val,
        X_val
    )

    return train_dataset, val_dataset