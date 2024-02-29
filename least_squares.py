import pandas as pd
import numpy as np
from leastsqdataset import SequenceDataset
import matplotlib.pyplot as plt
# import statsmodels.api as sm


#Read curated file, form a row of the A matrix[precip,precip_past12..............]


df = pd.read_csv('NHC7_curated_15m_7_past_12_test.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

target = 'level'
# features= ['precip', 'precip_past_12']
features= ['precip']

# # # Standardize the data Standardization does not change anything with least squares!
# target_mean = df[(df.valid) & (~(df.test))][target].mean()
# target_stdev = df[(df.valid) & (~(df.test))][target].std()

# for c in df.columns:
#     if (c == 'valid' or c == 'test'):
#         continue
#     mean = df[(df.valid) & (~(df.test))][c].mean() # This makes more sense than considering the added hourly zeros again for the mean calculation
#     stdev = df[(df.valid) & (~(df.test))][c].std()
#     df[c] = (df[c] - mean) / stdev

sequence_length = 28 #7

train_dataset = SequenceDataset(
    df,
    target=target,
    features=features,
    sequence_length=sequence_length,
    test = False
)
print(len(train_dataset))
test_dataset = SequenceDataset(
    df,
    target=target,
    features=features,
    sequence_length=sequence_length,
    test = True
)
# i = 0
# X, y = train_dataset[i]
# print(X)
# print(y)

A = []
b = []
for i in range(len(train_dataset)):
# for i in range(2):
    X, y = train_dataset[i]
    b += [y]
    A += [X.flatten()]

A = np.array(A)
b = np.array(b)

A = np.append(A,np.ones([len(A),1]),1)
# print(A)
w= np.linalg.lstsq(A,b, rcond=None)[0]
# print(w)

# axis = [i for i in range(len(w)-2,-1,-1)]
# plt.plot(axis,w[0:-1])
# # plt.xticks(np.arange(27,-1,-1))
# plt.xlim(max(axis),min(axis))
# plt.xticks(np.arange(0,len(w)-1,1))
# plt.show()

# Unstandardizing...
# b = b*target_stdev+target_mean
# pred = np.dot(A,w)*target_stdev+target_mean
# plt.plot(850-b)
# plt.plot(850 - pred)

# plt.plot(850-b)
# plt.plot(850 - np.dot(A,w))
# plt.show()

b_test = []
A_test = []
for i in range(len(test_dataset)):
    X, y = test_dataset[i]
    b_test += [y]
    A_test += [X.flatten()]

A_test = np.array(A_test)
b_test = np.array(b_test)

A_test = np.append(A_test,np.ones([len(A_test),1]),1)

# plt.plot(850-b_test)
# plt.plot(850 - np.dot(A_test,w))
# plt.show()

ystar_col = 'forecast'

df.loc[(df.valid)&~(df.test), ystar_col] = np.dot(A,w)
df.loc[(df.valid)&(df.test), ystar_col] = np.dot(A_test,w)
fig, ax1 = plt.subplots()
print(df)

ax1.plot(850 - df[(df.valid)&(df.test)][['level','forecast']]) # Test data
# ax1.plot(850 - df[(df.valid)&(~df.test)][['level','forecast']]) # Train data

plt.legend(["level", "prediction"],loc=4)
plt.ylabel("Stream height change (mm)")
plt.xticks(rotation=45)
# ax2=ax1.twiny()
# plt.twinx().plot(df[df.valid]['precip'], c='lightsteelblue', alpha = 0.9)
plt.twinx().plot(df['precip'], c='lightsteelblue', alpha = 0.9)
plt.gca().invert_yaxis()
plt.ylabel("Rainfall (mm)")
fig.tight_layout()
plt.show()