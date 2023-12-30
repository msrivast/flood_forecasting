import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


df_ws3 = pd.read_csv('WS3_final_2022.csv', usecols=['datetime', 'precip'])
df_ws3['datetime'] = pd.to_datetime(df_ws3['datetime'])

# df_ws3 = df_ws3[df_ws3["datetime"]>pd.Timestamp("2022-07-06 18:00")]
# df_ws3 = df_ws3[df_ws3["datetime"]<pd.Timestamp("2022-07-09 18:00")]

df_ws3['group'] = df_ws3.index 
df_ws3['count'] = df_ws3.set_index('datetime').resample('1min').bfill(limit=7)['group'].value_counts() #bfill() acts on originally missing data which it should not.
df_ws3['precip'] = df_ws3['precip']/df_ws3['count']
df_ws3= df_ws3.set_index('datetime').resample('1min').bfill(limit=7)
df_ws3= df_ws3.reset_index().drop(['group', 'count'], axis = 1)

df_ws3 = df_ws3.set_index("datetime")
df_ws3 = df_ws3.resample('1H', label = 'right', closed = 'right').sum(min_count=1) # writes 0 for no data -> 10/13 - 10/19 and no data before march
# df_ws3.to_csv("ws3.csv")
plt.plot(df_ws3)

# df_airport = pd.read_csv('oconee_2022.txt', usecols=['valid', 'precip_in'])
# df_airport['valid'] = pd.to_datetime(df_airport['valid'])
# df_airport = df_airport.set_index("valid")
# plt.plot(df_airport*25.4)

df_1697 = pd.read_csv('NHC1697_cleaned.csv', usecols=['edt', 'radardistance'])
df_1697['edt'] = pd.to_datetime(df_1697['edt']) # either the time to concentration is large or edt should be est

df_1697 = df_1697[df_1697["edt"]>pd.Timestamp("2022-01-01 00:00")]
df_1697 = df_1697[df_1697["edt"]<pd.Timestamp("2022-12-07 18:00")]
# df_1697 = df_1697[df_1697["edt"]>pd.Timestamp("2022-07-06 18:00")]
# df_1697 = df_1697[df_1697["edt"]<pd.Timestamp("2022-07-09 18:00")]

df_1697 = df_1697.set_index("edt")
# print(df_1697[df_1697.radardistance>2000])
df_1697[df_1697>2000] = np.NaN

df_1697 = df_1697.resample('1H', label = 'right', closed = 'right').min()
# df_1697.to_csv("NHC1697_hourly.csv")
plt.twinx().plot(1200-df_1697,c='magenta', alpha=0.7)
plt.title("WS#3 vs NHC_1697 Time to conc: 4-5hrs")
plt.ylabel("mm")
# plt.legend("NHC_1697")
# plt.savefig("all.pdf")
plt.show()