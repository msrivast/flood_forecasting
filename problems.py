import matplotlib.pyplot as plt

import numpy as np
import pandas as pd



df = pd.read_csv('curated_15m_7_past_12_test.csv')
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


# Divide train and test
test_start = "2022-09-15"

df_train = df.loc[:test_start].copy()
df_test = df.loc[test_start:].copy()

print("df train stats")
print(df_train[df_train.valid].describe())
print("----------")

print("df test stats")
print(df_test[df_test.valid].describe())
print("----------")
