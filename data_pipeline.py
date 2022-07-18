import torch, gzip, os, pandas as pd
import numpy as np
from torch.utils.data import Dataset

class EdfDataset(Dataset):
    def __init__(self, df, dataset_dir):
        self.csv = df
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        edf_path = os.path.join(self.dataset_dir, self.csv.iloc[idx, 0])        

        if self.dataset_dir=="./CHBMIT_1s_0.75OW/":
            f = gzip.GzipFile(edf_path, "r")
            signals = np.load(file=f).astype(np.float32)
            f.close()
        elif self.dataset_dir=="./bonn_256/":
            signals = pd.read_csv(edf_path, header=None).to_numpy(dtype=np.float32).T    
        elif self.dataset_dir=="./IIT_Delhi_256/IIT_Delhi_256/":
            signals = pd.read_csv(edf_path, header=None).to_numpy(dtype=np.float32).T
            
        signals = torch.from_numpy(signals)
        label = float(self.csv.iloc[idx, 1])
        
        return signals, label, edf_path
