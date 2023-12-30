# How should we train?

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


# Create a single DF with hourly* rainfall and hourly min() water level

df = pd.read_csv('ws2_hourly.csv')
df["level"] = pd.read_csv('NHC7_hourly.csv', usecols=['radardistance_mm'])
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')
# print(df)

# df = df[df["datetime"]>pd.Timestamp("2022-08-25 00:00")]
# df = df[df["datetime"]<pd.Timestamp("2022-09-01 00:00")] 
# df = df.set_index('datetime')
# # df_curated.to_csv('curated.csv')
# plt.plot(df['precip'])
# plt.twinx().plot(850 - df['level'], c = 'magenta', alpha = 0.7)
# plt.show()

# #Days where it rained

# df_test = df.groupby(pd.Grouper(key='datetime', axis=0, freq='D')).sum()
# df_test=df_test[df_test['precip']>3]['precip']
# print(df_test.count())

# Removing all hours which did not have a rain fall event in the past 6 hours.
# We chose 6 hours here because I wanted to provide a sequence length of 5,
# time to conc is 4hrs. R _ _ _ S. 
# Increasing the sequence length to 6 or 7 hours might be beneficial
# Also, this methind of getting 5 trailing zeros creates bad training data for
# all zero sequences, becuase the water level after each rain event is different and not representative of the
# start of the rainfall.

# df['precip_past_24'] = df['precip'].rolling(window=24,min_periods=1).sum()
df['can_delete'] = df['precip'].rolling(window=7).sum() #8/28 5PM rainfall makes the compelling case of increasing the post rainfall window to 6+2 hours 
# df['valid'] = df.can_delete != 0 # All valid sequence start points are marked
# df['can_delete'] = df['can_delete'].rolling(window=7,min_periods=1).sum().shift(-6)
# df=df.dropna()
df_curated = df[df.can_delete != 0].drop(['can_delete'], axis=1)
# df_curated = df

df_curated = df_curated.dropna()

# df_curated = df_curated[df_curated["datetime"]>pd.Timestamp("2022-08-01 00:00")]
# df_curated = df_curated[df_curated["datetime"]<pd.Timestamp("2022-09-05 00:00")] 

# df_curated = df_curated.set_index('datetime')

df_curated.to_csv('curated_7_past_24.csv',float_format='%.3f')
# df_curated.to_csv('curated_7_past_24.csv')
# df_curated.to_csv('test.csv')
# plt.plot(df_curated['precip'])
# plt.twinx().plot(850 - df_curated['level'], c = 'magenta', alpha = 0.7)
# plt.show()

