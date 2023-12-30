import torch
import numpy as np
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()
        # self.df = dataframe
        self.a = np.where(dataframe.valid)[0]
        # print(self.X.shape[0])
        # print(self.a.size)

    def __len__(self):
        # return self.X.shape[0]
        # return (self.df.valid).sum()
        return self.a.size

    def __getitem__(self, i): 
        # All rows are valid. Past data would need to be padded for each
        # isolated rain event. The past amount would need to vary based
        # on the length of the isolation window.

        # REPLICATE CURRENT BEHAVIOR
        # For every row bring bring the past rows that exist in the seq_length
        # time period padding with zeros as required

        if self.a[i] >= self.sequence_length - 1:
            i_start = self.a[i] - self.sequence_length + 1
            x = self.X[i_start:(self.a[i] + 1), :]
            # x = self.X[i_start:(i + 1)]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            # BUG padding = torch.tensor(0.0).repeat(self.sequence_length - self.a[i] - 1,1)
            x = self.X[0:(self.a[i] + 1), :]
            # x = self.X[0:(i + 1)]
            x = torch.cat((padding, x), 0)

        return x, self.y[self.a[i]]
        # return x, self.y[i].reshape(1,1)
        