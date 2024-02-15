# How should we train?

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


# Create a single DF with hourly* rainfall and hourly min() water level
# This file has a timestamp for every hour

df = pd.read_csv('ws2_hourly.csv')
df["level"] = pd.read_csv('NHC7_hourly.csv', usecols=['radardistance_mm'])
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')
# print(df)

# See note below
# df['precip_past_24'] = df['precip'].rolling(window='24H',min_periods=1).sum()
df['precip_past_12'] = df['precip'].rolling(window='12H',min_periods=1).sum()
df['can_delete'] = df['precip'].rolling(window=7,min_periods=1).sum() #7 hours after the end of rainfall
df['valid'] = df.can_delete != 0 # All valid sequence start points are marked
df['can_delete'] = df['can_delete'].rolling(window=7, min_periods=1).sum().shift(-6) #7 hours before the start of rainfall
# df=df.dropna(subset='can_delete')
df_curated = df[df.can_delete != 0].drop(['can_delete'], axis=1)
# df_curated = df

df_curated = df_curated.dropna()


# df_curated.to_csv('curated_7_past_12.csv',float_format='%.3f')

# df_curated.to_csv('curated_7_better.csv')
# df_curated.to_csv('test.csv')
# plt.plot(df_curated['precip'])
# plt.twinx().plot(850 - df_curated['level'], c = 'magenta', alpha = 0.7)
# plt.show()

a = np.where(df_curated['valid'])[0]
# r = np.random.choice(a, 1719, replace=False)
r = np.random.choice(a, 435, replace=False)

df_curated['test'] = False
df_curated.test.iloc[[r]] = True
df_curated.to_csv('curated_7_past_12_test.csv',float_format='%.3f')
print(df_curated)

# There are ~10 consecutive zeros at 8/03

# Another intersting instance ->
# 2022-02-21 11:00:00,0.000,,,True
# 2022-02-21 12:00:00,0.000,,,True
# 2022-02-21 13:00:00,0.000,,,True
# 2022-02-21 14:00:00,0.000,,,True
# 2022-02-21 15:00:00,0.000,,,True
# 2022-02-21 16:00:00,0.000,852.000,,True
# 2022-02-21 17:00:00,0.000,852.000,0.000,False
# 2022-02-21 18:00:00,0.333,852.000,0.333,True

# NOTE: This is good for now but this is vulnerable to WS and guage data becoming unavailable. The 0s and past 24h data
# are taken from the previous rain event when such anomalies occur. The best way to do it is to take hourly data of all rain events and
# write a new __getitem__(self, i) function that calculates the 24hr totals.