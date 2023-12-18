import numpy as np
from torch.utils.data import Dataset


class uGUIDEDataset(Dataset):
    def __init__(self, theta: np.ndarray, X: np.ndarray) -> None:
        super().__init__()

        self.theta = theta
        self.X = X

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, index):
        return self.theta[index], self.X[index]
    

def split_data(theta, X, val_ratio=0.05):
    n_data = len(theta)
    n_data_val = int(n_data * val_ratio)

    # torch data loaders can deal with the shufle for you, at each epoch, I wouldn't bother
    perm = np.random.permutation(n_data)

    theta_train = theta[perm][n_data_val:]
    theta_val = theta[perm][:n_data_val]

    X_train = X[perm][n_data_val:]
    X_val = X[perm][:n_data_val]

    train_dataset = uGUIDEDataset(
        theta_train,
        X_train
    )
    val_dataset = uGUIDEDataset(
        theta_val,
        X_val
    )

    return train_dataset, val_dataset