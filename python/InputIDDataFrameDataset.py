from torch.utils.data import Dataset
import pandas as pd
import torch

class InputIDDataFrameDataset(Dataset):
    def __init__(self, df):
        self.df = df      

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):        
        input_ids, label = self.df[['input_ids', 'label']].iloc[idx, :]     
        
        return input_ids, label
