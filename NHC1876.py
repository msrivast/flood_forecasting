import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# df_ws2 = pd.read_csv('WS2_final_2022.csv', usecols=['datetime', 'precip'])
# df_ws2['datetime'] = pd.to_datetime(df_ws2['datetime'])
# df_ws2 = df_ws2.set_index("datetime")
# df_ws2 = df_ws2.resample('1H', label = 'right', closed = 'right').sum(min_count=1) # writes 0 for no data -> 7/28 - 8/03 
# # df_ws2.to_csv("ws2.csv")
# plt.plot(df_ws2)

# df_ws3 = pd.read_csv('WS3_final_2022.csv', usecols=['datetime', 'precip'])
# df_ws3['datetime'] = pd.to_datetime(df_ws3['datetime'])
# df_ws3 = df_ws3.set_index("datetime")
# # df_ws3 = df_ws3.resample('1H', label = 'right', closed = 'right').sum(min_count=1) # writes 0 for no data -> 10/13 - 10/19 and no data before march
# # df_ws3.to_csv("ws3.csv")
# plt.plot(df_ws3)

# df_airport = pd.read_csv('oconee_2022.txt', usecols=['valid', 'precip_in'])
# df_airport['valid'] = pd.to_datetime(df_airport['valid'])
# df_airport = df_airport.set_index("valid")
# plt.plot(df_airport*25.4)

df_1876 = pd.read_csv('NHC1876_2022.csv', usecols=['datetime', 'r_median_dist']) # 6-7 minute data!!!!!!!
df_1876['datetime'] = pd.to_datetime(df_1876['datetime']) # either the time to concentration is large or edt should be est

df_1876 = df_1876[df_1876["datetime"]>pd.Timestamp("2022-01-01 00:00")]
df_1876 = df_1876[df_1876["datetime"]<pd.Timestamp("2022-11-21 00:00")] 

df_1876 = df_1876.set_index("datetime")

# df_1876 = df_1876.resample('1H', label = 'right', closed = 'right').mean()
# df_1876.to_csv("NHC1876_hourly.csv")
plt.twinx().plot(df_1876,c='magenta', alpha=0.7)
plt.title("WS#2 vs NHC_1876")
plt.ylabel("mm")
# plt.legend("NHC_1876")
# plt.savefig("all.pdf")
plt.show()