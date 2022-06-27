import torch, gzip, time, os, pandas as pd
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset

class EdfDataset(Dataset):
    def __init__(self, df, dataset_dir, transform = None, target_transform=None):
        self.csv = df
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        edf_path = os.path.join(self.dataset_dir, self.csv.iloc[idx, 0])        

        signals = pd.read_csv(edf_path, header=None).to_numpy(dtype=np.float32).T
        signals = torch.from_numpy(signals)
        signals = signals.unsqueeze(0)
        label = float(self.csv.iloc[idx, 1])
        
        if self.transform:
            signals = self.transform(signals)

        if self.target_transform:
            label = self.target_transform(label)
        
        return signals, label, edf_path