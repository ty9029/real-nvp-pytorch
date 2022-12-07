import numpy as np
from torch.utils.data import Dataset


class Normal(Dataset):
    def __init__(self, num_data):
        self.num_data = num_data
        self.data = np.array(np.random.normal(0, 1.0, (num_data, 2)), dtype="float32")

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return self.data[idx, :]


class DoubleMoons(Dataset):
    def __init__(self, num_data):
        self.num_data = num_data

        self.init_data(num_data)

    def init_data(self, n_samples):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out

        outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out)) - 0.5
        outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out)) - 0.1
        inner_circ_x = 0.5 - np.cos(np.linspace(0, np.pi, n_samples_in))
        inner_circ_y = 0.1 - np.sin(np.linspace(0, np.pi, n_samples_in))

        X = np.vstack([np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]).T
        X += np.random.normal(0, 0.05, X.shape)

        self.data = X.astype("float32")

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        return self.data[idx, :]
