import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str, args) -> None:
        self.split = split
        self.data_dir = data_dir
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))

        # スケーリング
        self.X = (self.X - self.X.mean(dim=0)) / self.X.std(dim=0)

        # ベースライン補正
        baseline = self.X[:, :, :100].mean(dim=2, keepdim=True)
        self.X = self.X - baseline

        self.num_classes = len(torch.unique(self.y))
        self.seq_len = self.X.shape[2]
        self.num_channels = self.X.shape[1]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

