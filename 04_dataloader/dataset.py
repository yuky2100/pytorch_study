# dataset.py
import torch
from torch.utils.data import Dataset

class XORDataset(Dataset):
    def __init__(self):
        # データ定義（4組）
        self.x_data = torch.tensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])
        self.y_data = torch.tensor([
            [0.0],
            [1.0],
            [1.0],
            [0.0]
        ])
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]
