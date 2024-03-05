import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

df_ws2 = pd.read_csv('WS2_final_2022.csv', usecols=['datetime', 'precip'])
df_ws2['datetime'] = pd.to_datetime(df_ws2['datetime'])

# df_ws2 = df_ws2[df_ws2["datetime"]>pd.Timestamp("2022-5-24 18:00")]
# df_ws2 = df_ws2[df_ws2["datetime"]<pd.Timestamp("2022-5-28 00:00")] 

# df_ws2 = df_ws2.set_index("datetime")
# df_ws2[df_ws2<0.1] = 0
# df_ws2 = df_ws2.reset_index()

df_ws2['group'] = df_ws2.index
df_ws2['count'] = df_ws2.set_index('datetime').resample('1min').bfill(limit=7)['group'].value_counts() # bfill() writes 0 for no data -> 7/22 - 8/03 - > manually deleted data between 7/22 to 7/28 
df_ws2          = df_ws2.set_index('datetime').resample('1min').bfill(limit=7)
df_ws2          = df_ws2.div(df_ws2['count'], axis = 0).reset_index().drop(['group','count'], axis = 1)
df_ws2 = df_ws2.set_index("datetime")
df_ws2 = df_ws2.resample('15T', label = 'right', closed = 'right').sum(min_count=1)

df_ws2[df_ws2<0.2] = 0

# df_ws2.to_csv("ws2_10Min.csv")
plt.plot(df_ws2)
# plt.plot(df_ws2, c = 'cyan', alpha=0.7)
plt.ylabel("Rainfall (mm)")
# plt.show()

# df_7 = pd.read_csv('NHC7_cleaned.csv', usecols=['edt', 'radardistance_mm']) # 6-7 minute data!!!!!!!
# df_7['edt'] = pd.to_datetime(df_7['edt']) # either the time to concentration is large or edt should be est

df_7 = pd.read_csv('NHC296_final.csv', usecols=['datetime', 'r_median_dist']) # 12 minute data!!!!!!!
df_7.rename(columns={'datetime':'edt', 'r_median_dist':'radardistance_mm'},inplace=True)
df_7['edt'] = pd.to_datetime(df_7['edt']) # either the time to concentration is large or edt should be est


# df_7 = pd.read_csv('NHC1697_cleaned.csv', usecols=['edt', 'radardistance']) # 12 minute data!!!!!!!
# df_7.rename(columns={'radardistance':'radardistance_mm'},inplace=True)
# df_7['edt'] = pd.to_datetime(df_7['edt']) # either the time to concentration is large or edt should be est

# df_7 = df_7[df_7["edt"]>pd.Timestamp("2022-01-01 00:00")]
# df_7 = df_7[df_7["edt"]<pd.Timestamp("2022-11-21 00:00")]
# df_7 = df_7[df_7["edt"]>pd.Timestamp("2022-5-24 18:00")]
# df_7 = df_7[df_7["edt"]<pd.Timestamp("2022-5-28 00:00")] 
df_7 = df_7[df_7["edt"]>pd.Timestamp("2022-01-01 00:00")]
df_7 = df_7[df_7["edt"]<pd.Timestamp("2022-12-07 00:00")]

df_7 = df_7.set_index("edt")
# plt.twinx().plot(855-df_7, c='green', alpha = 0.7)

# df_7 = df_7.resample('10T', label = 'right', closed = 'right').mean()
# df_7 = df_7.resample('1T').interpolate().resample('30T').asfreq()
# df = df_7
df_7[df_7<700] = np.NaN
df_7[df_7>1100] = np.NaN
# df_7[df_7>2000] = np.NaN
df_7 = df_7.resample('15T', label = 'right', closed = 'right').min()
# df_7.to_csv("NHC7_30min.csv")
# plt.twinx().plot(855-df_7,c='magenta', alpha=0.7)
# plt.twinx().plot(1050-df_7,c='magenta', alpha=0.7)
# plt.plot(855-df, alpha=0.7)
# plt.title("WS#2(15min) vs NHC_7(min) 4-hr time to concentration(May)")

# plt.title("WS#2(15min) vs NHC_296(min) 4-hr time to concentration(May)")
# plt.ylabel("Level rise (mm)")
# plt.legend("NHC_7")
# plt.savefig("hourly.pdf")
df = df_ws2
df['level'] = df_7['radardistance_mm']
# plt.show()

df['precip_past_12'] = df['precip'].rolling(window='12H',min_periods=1).sum()
df['can_delete'] = df['precip'].rolling(window=28,min_periods=1).sum() #7 hours after the end of 15-minute rainfall
df['valid'] = df.can_delete != 0 # All valid sequence start points are marked
df['can_delete'] = df['can_delete'].rolling(window=28, min_periods=1).sum().shift(-27) #7 hours before the start of 15-minute rainfall
# df=df.dropna(subset='can_delete')
# print(df)
# df.to_csv('test.csv',float_format='%.0f')
df_curated = df[df.can_delete != 0]
# df_curated = df

df_curated = df_curated.dropna().drop(['can_delete'], axis=1)

# print(df_curated)
# df_curated.to_csv('curated_15m_7_past_12.csv',float_format='%.3f')

# Code to create a boolean column marking test rows

a = np.where(df_curated['valid'])[0]
# r = np.random.choice(a, 1719, replace=False)
# r = np.random.choice(a, 180, replace=False) #608 for 7 hours and >0.2 rainfall
# r = np.random.choice(a, 650, replace=False)
r = np.random.choice(a, 620, replace=False)
# r = np.random.choice(a, 614, replace=False)
# r = np.random.choice(a, 425, replace=False)

df_curated['test'] = False
df_curated.test.iloc[[r]] = True

df = df.dropna()
plt.twinx().plot(1100-df['level'],c='magenta', alpha=0.7)
plt.twinx().plot(1100-df_curated[df_curated.valid]['level'], c='green', alpha=0.7)
plt.title("WS#2(15min) vs NHC_296(min) 4-hr time to concentration(May)")
plt.ylabel("Level rise (mm)")
plt.show()

# df_curated.to_csv('NHC296_curated_15m_7_past_12_test_low_noise.csv',float_format='%.0f')
print(df_curated)