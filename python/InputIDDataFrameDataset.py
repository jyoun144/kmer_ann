from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np

class InputIDDataFrameDataset(Dataset):
    def __init__(self, df):
        self.df = df      

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):        
        input_ids, label = self.df[['input_ids', 'label']].iloc[idx, :]     
        
        return np.array(input_ids), label
