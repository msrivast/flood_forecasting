import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

df_ws2 = pd.read_csv('WS2_final_2022.csv', usecols=['datetime', 'precip'])
df_ws2['datetime'] = pd.to_datetime(df_ws2['datetime'])

# df_ws2 = df_ws2[df_ws2["datetime"]>pd.Timestamp("2022-5-24 18:00")]
# df_ws2 = df_ws2[df_ws2["datetime"]<pd.Timestamp("2022-5-28 00:00")] 

df_ws2['group'] = df_ws2.index
df_ws2['count'] = df_ws2.set_index('datetime').resample('1min').bfill(limit=7)['group'].value_counts() # bfill() writes 0 for no data -> 7/22 - 8/03 - > manually deleted data between 7/22 to 7/28 
df_ws2          = df_ws2.set_index('datetime').resample('1min').bfill(limit=7)
df_ws2          = df_ws2.div(df_ws2['count'], axis = 0).reset_index().drop(['group','count'], axis = 1)
df_ws2 = df_ws2.set_index("datetime")
df_ws2 = df_ws2.resample('15T', label = 'right', closed = 'right').sum(min_count=1)
# df_ws2.to_csv("ws2_10Min.csv")
plt.plot(df_ws2)
# plt.plot(df_ws2, c = 'cyan', alpha=0.7)
plt.ylabel("Rainfall (mm)")
# plt.show()

df_7 = pd.read_csv('NHC7_cleaned.csv', usecols=['edt', 'radardistance_mm']) # 6-7 minute data!!!!!!!
df_7['edt'] = pd.to_datetime(df_7['edt']) # either the time to concentration is large or edt should be est

# df_7 = df_7[df_7["edt"]>pd.Timestamp("2022-01-01 00:00")]
# df_7 = df_7[df_7["edt"]<pd.Timestamp("2022-11-21 00:00")]
# df_7 = df_7[df_7["edt"]>pd.Timestamp("2022-5-24 18:00")]
# df_7 = df_7[df_7["edt"]<pd.Timestamp("2022-5-28 00:00")] 

df_7 = df_7.set_index("edt")
# plt.twinx().plot(855-df_7, c='green', alpha = 0.7)

# df_7 = df_7.resample('10T', label = 'right', closed = 'right').mean()
# df_7 = df_7.resample('1T').interpolate().resample('30T').asfreq()
# df = df_7
df_7 = df_7.resample('15T', label = 'right', closed = 'right').min()
# df_7.to_csv("NHC7_30min.csv")
plt.twinx().plot(855-df_7,c='magenta', alpha=0.7)
# plt.plot(855-df, alpha=0.7)
plt.title("WS#2(15min) vs NHC_7(min) 4-hr time to concentration(May)")
plt.ylabel("Level rise (mm)")
# plt.legend("NHC_7")
# plt.savefig("hourly.pdf")
df = df_ws2
df['level'] = df_7['radardistance_mm']
plt.show()

df['precip_past_12'] = df['precip'].rolling(window='12H',min_periods=1).sum()
df['can_delete'] = df['precip'].rolling(window=28,min_periods=1).sum() #7 hours after the end of 15-minute rainfall
df['valid'] = df.can_delete != 0 # All valid sequence start points are marked
df['can_delete'] = df['can_delete'].rolling(window=28, min_periods=1).sum().shift(-27) #7 hours before the start of 15-minute rainfall
# df=df.dropna(subset='can_delete')
df_curated = df[df.can_delete != 0].drop(['can_delete'], axis=1)
# df_curated = df

df_curated = df_curated.dropna()

print(df_curated)
df_curated.to_csv('curated_15m_7_past_12.csv',float_format='%.3f')