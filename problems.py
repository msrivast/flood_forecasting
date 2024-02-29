import matplotlib.pyplot as plt

import numpy as np
import pandas as pd



df = pd.read_csv('NHC7_curated_15m_7_past_12_test.csv')
# df = pd.read_csv('curated_7_past_12_test.csv')
df = df.set_index('datetime')

print("df stats")
print(df[(df.valid)].describe())
print("----------")

print("df random train stats")
print(df[(df.valid)&(~df.test)].describe())
print("----------")

print("df random test stats")
print(df[df.valid & df.test].describe())
print("----------")

dfp = pd.read_csv('predictions.csv')
dfp = dfp.set_index('datetime')
# target_mean = df[(df.valid)&(~df.test)]['level'].mean()
target_mean = dfp[~dfp.test]['level'].mean()
# print(target_mean)
# target_std = df[(df.valid)&(~df.test)]['level'].std()
target_std = dfp[~dfp.test]['level'].std()
print(target_std)
dfp_test = (dfp[dfp.test]-target_mean)/target_std
print(dfp_test)
dfp_train = (dfp[~dfp.test]-target_mean)/target_std
# dfp_train.drop(['test'], axis=1)
# dfp_test.drop(['test'], axis=1)
# print("test MSE: ")
print("train MSE: ", (np.square(dfp_train.level - dfp_train.forecast)).sum()/len(dfp_train.index))
print("test MSE: ", (np.square(dfp_test.level - dfp_test.forecast)).sum()/len(dfp_test.index))
# print(len(dfp_test.index))
# print(dfp)

# df_out = dfp[['level','forecast','test']]

# for c in df_out.columns:
#     df_out[c] = df_out[c] * target_std + target_mean

# print(df_out)

# # Divide train and test
# test_start = "2022-09-15"

# df_train = df.loc[:test_start].copy()
# df_test = df.loc[test_start:].copy()

# print("df train stats")
# print(df_train[df_train.valid].describe())
# print("----------")

# print("df test stats")
# print(df_test[df_test.valid].describe())
# print("----------")
