import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('Using device:', device)
from torch.utils.data import DataLoader
from torch import nn
from dataset_test_randomized import SequenceDataset
from neuralnet import ShallowRegressionLSTM
# from soft_dtw_cuda import SoftDTW


df = pd.read_csv('curated_7_past_12_test.csv')
df = df.set_index('datetime')

# # Divide train and test
# test_start = "2022-09-15"

# df_train = df.loc[:test_start].copy()
# df_test = df.loc[test_start:].copy()
# print(df_test)
# print("Total sequences:", df.valid.sum())
# print("Total test sequences:", df_test.valid.sum())
# print("Test set fraction:", df_test.valid.sum() / df.valid.sum())


target = 'level'
# target = 'level_diff'
# features= ['precip', 'precip_past_24']
features= ['precip', 'precip_past_12']

# # Standardize the data
target_mean = df[(df.valid) & (~(df.test))][target].mean()
target_stdev = df[(df.valid) & (~(df.test))][target].std()
# precip_mean = df_train[df_train.valid]['precip'].mean()
# precip_stdev = df_train[df_train.valid]['precip'].std()

for c in df.columns:
    if (c == 'valid' or c == 'test'):
        continue
    # if (c == target): #
    mean = df[(df.valid) & (~(df.test))][c].mean() # This makes more sense than considering the added hourly zeros again for the mean calculation
    stdev = df[(df.valid) & (~(df.test))][c].std()
    # else:
    #     mean = df_train[c].mean()
    #     stdev = df_train[c].std()
    
    # df_train[c] = (df_train[c] - mean) / stdev
    # df_test[c] = (df_test[c] - mean) / stdev
    df[c] = (df[c] - mean) / stdev

# print(df_test)

i = 25
sequence_length = 7 #7

train_dataset = SequenceDataset(
    df,
    target=target,
    features=features,
    sequence_length=sequence_length,
    test = False
)

test_dataset = SequenceDataset(
    df,
    target=target,
    features=features,
    sequence_length=sequence_length,
    test = True
)

# X, y = train_dataset[i]
# print(X)
# print(y)

torch.manual_seed(99)
batch_size=64
# batch_size=4

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False) #in this randomized version, they are already separated/shuffled


X, y = next(iter(train_loader))
print("Features shape:", X.shape)
print("Target shape:", y.shape)


learning_rate = 5e-5
num_hidden_units = 16

model = ShallowRegressionLSTM(num_features=len(features), hidden_units=num_hidden_units).to(device)
loss_function = nn.MSELoss()
# loss_function = SoftDTW(gamma=1.0, normalize=True) # just like nn.MSELoss()
# loss_function = SoftDTW(use_cuda=False, gamma=1.0, normalize=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = loss_function(output, y)
        # loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        # total_loss += loss

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss

def test_model(data_loader, model, loss_function):
    
    num_batches = len(data_loader)
    # print(num_batches)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            # print(y*target_stdev + target_mean,output*target_stdev + target_mean)
            # print(X*precip_stdev + precip_mean)
            # total_loss += loss.item()
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss

print("Untrained test\n--------")
test_model(test_loader, model, loss_function)
print()

for ix_epoch in range(2000):
    print(f"Epoch {ix_epoch}\n---------")
    train_loss = train_model(train_loader, model, loss_function, optimizer=optimizer)
    test_loss = test_model(test_loader, model, loss_function)
    print()
    # if(train_loss - test_loss <= 0.01): # insufficient improvement
    # if(test_loss - train_loss >= 0.09): # 0.15 looks like a lot, hesitate to say overfit(with 32N*2layers, training loss ~0.04)
        # The 12 hour past data needs 0.09 becuase the test loss is pretty small to begin with -> suggesting 12h is a good choice!
    # if(train_loss < 0.1):
        # break


def predict(data_loader, model):

    output = torch.tensor([],device=device)
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            X = X.to(device)
            y_star = model(X)
            # output = torch.cat((output, y_star.flatten()), 0)
            output = torch.cat((output, y_star), 0)
    
    return output


train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

ystar_col = "forecast"

# Discard the rows that were just used in the training
# df_train = df_train[df_train.valid]
# df_test = df_test[df_test.valid]
df = df[df.valid]

df.loc[~(df.test), ystar_col] = predict(train_eval_loader, model).cpu().numpy()
df.loc[(df.test), ystar_col] = predict(test_loader, model).cpu().numpy()

# print(df.iloc[2971])

df_out = df[['level',ystar_col,'test']]

for c in df_out.columns:
    if (c == 'test'):
        continue
    df_out[c] = df_out[c] * target_stdev + target_mean

# print(df_out.iloc[2971])

# # print(df_out[df_out.test])
df_out.to_csv('predictions.csv', float_format='%.0f')

# print("Full dataset: ")
# print("-------------")
res = df_out['level'].sub(df_out['forecast']).pow(2).sum()
tot = df_out['level'].sub(df_out['level'].mean()).pow(2).sum()
r2 = 1 - res/tot
print("R2 FULL: ", r2)

# print("Train dataset: ")
# print("-------------")
res = df_out[~df_out.test]['level'].sub(df_out[~df_out.test]['forecast']).pow(2).sum()
tot = df_out[~df_out.test]['level'].sub(df_out[~df_out.test]['level'].mean()).pow(2).sum()
r2 = 1 - res/tot
print("R2 TRAIN: ", r2)

# print("Test dataset: ")
# print("-------------")
res = df_out[df_out.test]['level'].sub(df_out[df_out.test]['forecast']).pow(2).sum()
tot = df_out[df_out.test]['level'].sub(df_out[df_out.test]['level'].mean()).pow(2).sum()
r2 = 1 - res/tot
print("R2 TEST: ", r2)

plt.plot(850-df_out['level'])
plt.plot(850-df_out['forecast'], alpha = 0.7)
# plt.plot(850-df_out[df_out.test]['level'])
# plt.plot(850-df_out[df_out.test]['forecast'], alpha = 0.7)
plt.title("Water level vs prediction(mm)")
plt.legend(["level", "prediction"])
plt.ylabel("mm")
plt.show()


