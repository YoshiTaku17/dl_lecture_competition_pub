import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = self.load_npy_files(os.path.join(data_dir, f"{split}_X"))
        self.subject_idxs = self.load_npy_files(os.path.join(data_dir, f"{split}_subject_idxs"))
        
        if split in ["train", "val"]:
            self.y = self.load_npy_files(os.path.join(data_dir, f"{split}_y"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
        
        # 標準化を追加
        self.X = (self.X - self.X.mean(dim=0)) / self.X.std(dim=0)

    def load_npy_files(self, dir_path):
        npy_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.npy')]
        npy_files.sort()  # Ensure the files are loaded in order
        data = [np.load(f) for f in npy_files]
        return torch.tensor(np.stack(data))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
