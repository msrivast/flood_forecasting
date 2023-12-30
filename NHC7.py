import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

df_ws2 = pd.read_csv('WS2_final_2022.csv', usecols=['datetime', 'precip'])
df_ws2['datetime'] = pd.to_datetime(df_ws2['datetime'])

# df_ws2 = df_ws2[df_ws2["datetime"]>pd.Timestamp("2022-7-5 18:00")]
# df_ws2 = df_ws2[df_ws2["datetime"]<pd.Timestamp("2022-7-15 00:00")] 

df_ws2['group'] = df_ws2.index 
df_ws2['count'] = df_ws2.set_index('datetime').resample('1min').bfill(limit=7)['group'].value_counts() #bfill() acts on originally missing data which it should not.
df_ws2['precip'] = df_ws2['precip']/df_ws2['count']
df_ws2= df_ws2.set_index('datetime').resample('1min').bfill(limit=7)
df_ws2= df_ws2.reset_index().drop(['group', 'count'], axis = 1)

df_ws2 = df_ws2.set_index("datetime")
df_ws2 = df_ws2.resample('1H', label = 'right', closed = 'right').sum(min_count=1) 
# df_ws2.to_csv("ws2_hourly.csv")
plt.plot(df_ws2)
plt.ylabel("Rainfall (mm)")
# df_ws3 = pd.read_csv('WS3_final_2022.csv', usecols=['datetime', 'precip'])
# df_ws3['datetime'] = pd.to_datetime(df_ws3['datetime'])
# df_ws3 = df_ws3.set_index("datetime")
# # df_ws3 = df_ws3.resample('1H', label = 'right', closed = 'right').sum(min_count=1) # writes 0 for no data -> 10/13 - 10/19 and no data before march
# # df_ws3.to_csv("ws3.csv")
# plt.plot(df_ws3, alpha = 1)

# df_airport = pd.read_csv('oconee_2022.txt', usecols=['valid', 'precip_in'])
# df_airport['valid'] = pd.to_datetime(df_airport['valid'])
# df_airport = df_airport.set_index("valid")
# plt.plot(df_airport*25.4)

df_7 = pd.read_csv('NHC7_cleaned.csv', usecols=['edt', 'radardistance_mm']) # 6-7 minute data!!!!!!!
df_7['edt'] = pd.to_datetime(df_7['edt']) # either the time to concentration is large or edt should be est

df_7 = df_7[df_7["edt"]>pd.Timestamp("2022-01-01 00:00")]
df_7 = df_7[df_7["edt"]<pd.Timestamp("2022-11-21 00:00")]
# df_7 = df_7[df_7["edt"]>pd.Timestamp("2022-07-05 18:00")]
# df_7 = df_7[df_7["edt"]<pd.Timestamp("2022-07-15 00:00")] 

df_7 = df_7.set_index("edt")

df_7 = df_7.resample('1H', label = 'right', closed = 'right').min()
# df_7.to_csv("NHC7_hourly.csv")
plt.twinx().plot(855-df_7,c='magenta', alpha=0.4)
plt.title("WS#2 vs NHC_7")
plt.ylabel("Level rise (mm)")
# plt.legend("NHC_7")
# plt.savefig("all.pdf")
plt.show()