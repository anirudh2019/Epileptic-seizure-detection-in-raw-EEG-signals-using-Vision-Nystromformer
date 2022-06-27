import torch, gzip, time, os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

        f = gzip.GzipFile(edf_path, "r")
        signals = np.load(file=f).astype(np.float32)
        signals = torch.from_numpy(signals)
        f.close()

        signals = signals.unsqueeze(0)
        label = float(self.csv.iloc[idx, 1])
        
        if self.transform:
            signals = self.transform(signals)

        if self.target_transform:
            label = self.target_transform(label)
        
        return signals, label, edf_path
