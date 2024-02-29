import numpy as np

class SequenceDataset():
    def __init__(self, dataframe, target, features, sequence_length=5, test = False):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = dataframe[target].values
        self.X = dataframe[features].values
        # self.df = dataframe
        if (test == True):
            self.a = np.where((dataframe.valid) & (dataframe.test))[0]
        else:
            self.a = np.where((dataframe.valid) & (~(dataframe.test)))[0]

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
            # padding = self.X[0].repeat(self.sequence_length - self.a[i] - 1, 1) #added a row of zero entries to the curated file
            padding = np.repeat([self.X[0]], self.sequence_length - self.a[i] - 1,axis=0)
            # BUG padding = torch.tensor(0.0).repeat(self.sequence_length - self.a[i] - 1,1) #can't add zeros to standardized data!
            x = self.X[0:(self.a[i] + 1), :]
            # x = self.X[0:(i + 1)]
            x = np.concatenate((padding, x), 0)

        return x, self.y[self.a[i]]
        # return x, self.y[i].reshape(1,1)