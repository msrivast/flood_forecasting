# # Want to control the datetime x-tick labels
# import matplotlib.pyplot as plt

# import numpy as np
# import pandas as pd

# import datetime
# import matplotlib.units as munits
# import matplotlib.dates as mdates

# converter = mdates.ConciseDateConverter()
# munits.registry[np.datetime64] = converter
# munits.registry[datetime.date] = converter
# munits.registry[datetime.datetime] = converter

# # array = np.loadtxt('oconee_2022.txt',delimiter=',', skiprows = 2143, usecols=[2, 3])

# # Show time on the x-axis

# df_airport = pd.read_csv('oconee_2022.txt', usecols=['valid', 'precip_in'])
# df_airport['valid'] = pd.to_datetime(df_airport['valid'])
# df_airport = df_airport[df_airport["valid"]<pd.Timestamp("2022-12-07 18:00")]
# # print(df.valid.dtype)
# # print(df.head())
# # plt.plot(df_airport['valid'], df_airport['precip_in']*25.4)

# df_airport = df_airport.set_index("valid")
# # df_airport.plot()
# f = plt.plot(df_airport*25.4,':')

# plt.show()





#####################################
#Difference between binning independent data and distributing across the time interval and summing till the desired interval!

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# df_ws2 = pd.read_csv('WS2_final_2022.csv', usecols=['datetime', 'precip'])
# df_ws2['datetime'] = pd.to_datetime(df_ws2['datetime'])

# df_ws2 = df_ws2[df_ws2["datetime"]>pd.Timestamp("2022-5-24 18:00")]
# df_ws2 = df_ws2[df_ws2["datetime"]<pd.Timestamp("2022-5-28 00:00")] 

# # df_ws2['group'] = df_ws2.index
# # df_ws2['count'] = df_ws2.set_index('datetime').resample('1min').bfill(limit=7)['group'].value_counts() # bfill() writes 0 for no data -> 7/22 - 8/03 - > manually deleted data between 7/22 to 7/28 
# # df_ws2          = df_ws2.set_index('datetime').resample('1min').bfill(limit=7)
# # df_ws2          = df_ws2.div(df_ws2['count'], axis = 0).reset_index().drop(['group','count'], axis = 1)
# df_ws2 = df_ws2.set_index("datetime")
# # df_ws2 = df_ws2.resample('10T', label = 'right', closed = 'right').sum(min_count=1)
# # df_ws2.to_csv("ws2_10Min.csv")
# plt.plot(df_ws2)

df_ws2 = pd.read_csv('WS2_final_2022.csv', usecols=['datetime', 'precip'])
df_ws2['datetime'] = pd.to_datetime(df_ws2['datetime'])

df_ws2 = df_ws2[df_ws2["datetime"]>pd.Timestamp("2022-5-24 18:00")]
df_ws2 = df_ws2[df_ws2["datetime"]<pd.Timestamp("2022-5-28 00:00")] 

df_ws2['group'] = df_ws2.index
df_ws2['count'] = df_ws2.set_index('datetime').resample('1min').bfill(limit=7)['group'].value_counts() # bfill() writes 0 for no data -> 7/22 - 8/03 - > manually deleted data between 7/22 to 7/28 
df_ws2          = df_ws2.set_index('datetime').resample('1min').bfill(limit=7)
df_ws2          = df_ws2.div(df_ws2['count'], axis = 0).reset_index().drop(['group','count'], axis = 1)
df_ws2 = df_ws2.set_index("datetime")
df_ws2 = df_ws2.resample('60T', label = 'right', closed = 'right').sum(min_count=1)
# df_ws2.to_csv("ws2_10Min.csv")
plt.plot(df_ws2)
# plt.plot(df_ws2, c = 'cyan', alpha=0.7)
plt.ylabel("Rainfall (mm)")
# plt.show()

# # df_ws2 = pd.read_csv('WS2_final_2022.csv', usecols=['datetime', 'precip'])
# # df_ws2['datetime'] = pd.to_datetime(df_ws2['datetime'])
# # # df_ws2['group'] = df_ws2.index
# # # df_ws2['count'] = df_ws2.set_index('datetime').resample('1min').bfill(limit=10)['group'].value_counts() # bfill() writes 0 for no data -> 7/22 - 8/03 - > manually deleted data between 7/22 to 7/28 
# # # df_ws2          = df_ws2.set_index('datetime').resample('1min').bfill(limit=10)
# # # df_ws2          = df_ws2.div(df_ws2['count'], axis = 0).reset_index().drop(['group','count'], axis = 1)
# # df_ws2 = df_ws2.set_index("datetime")
# # df_ws2 = df_ws2.resample('10T', label = 'right', closed = 'right').sum(min_count=1)
# # # df_ws2.to_csv("ws2_10Min.csv")
# # plt.plot(df_ws2, alpha = 0.7)

# # df_ws3 = pd.read_csv('WS3_final_2022.csv', usecols=['datetime', 'precip'])
# # df_ws3['datetime'] = pd.to_datetime(df_ws3['datetime'])
# # df_ws3 = df_ws3.set_index("datetime")
# # # df_ws3 = df_ws3.resample('1H', label = 'right', closed = 'right').sum(min_count=1) # writes 0 for no data -> 10/13 - 10/19 and no data before march
# # # df_ws3.to_csv("ws3.csv")
# # plt.plot(df_ws3, alpha = 1)

# # df_airport = pd.read_csv('oconee_2022.txt', usecols=['valid', 'precip_in'])
# # df_airport['valid'] = pd.to_datetime(df_airport['valid'])
# # df_airport = df_airport.set_index("valid")
# # plt.plot(df_airport*25.4)

df_7 = pd.read_csv('NHC7_cleaned.csv', usecols=['edt', 'radardistance_mm']) # 6-7 minute data!!!!!!!
df_7['edt'] = pd.to_datetime(df_7['edt']) # either the time to concentration is large or edt should be est

# df_7 = df_7[df_7["edt"]>pd.Timestamp("2022-01-01 00:00")]
# df_7 = df_7[df_7["edt"]<pd.Timestamp("2022-11-21 00:00")]
df_7 = df_7[df_7["edt"]>pd.Timestamp("2022-5-24 18:00")]
df_7 = df_7[df_7["edt"]<pd.Timestamp("2022-5-28 00:00")] 

df_7 = df_7.set_index("edt")
# plt.twinx().plot(855-df_7, c='green', alpha = 0.7)

# df_7 = df_7.resample('10T', label = 'right', closed = 'right').mean()
# df_7 = df_7.resample('1T').interpolate().resample('30T').asfreq()
df_7 = df_7.resample('60T', label = 'right', closed = 'right').min()
# df_7.to_csv("NHC7_30min.csv")
plt.twinx().plot(855-df_7,c='magenta', alpha=0.7)
plt.title("WS#2(hourly*) vs NHC_7(min) 4-hr time to concentration(May)")
plt.ylabel("Level rise (mm)")
# plt.legend("NHC_7")
# plt.savefig("hourly.pdf")
plt.show()

# import matplotlib.pyplot as plt

# import numpy as np
# import pandas as pd

# df_ws2 = pd.read_csv('test.csv', usecols=['datetime', 'precip'])
# df_ws2['datetime'] = pd.to_datetime(df_ws2['datetime'])

# df_ws2 = df_ws2[df_ws2["datetime"]>pd.Timestamp("2022-5-25 00:00")]
# df_ws2 = df_ws2[df_ws2["datetime"]<pd.Timestamp("2022-5-31 00:00")] 

# # Create unique group IDs by simply using the existing index (Assumes an integer, non-duplicated index)
# df_ws2['group'] = df_ws2.index  

# # Get the count of intervals for each post-resampled timestamp.
# df_ws2['count'] = df_ws2.set_index('datetime').resample('1min').bfill(limit=10)['group'].value_counts()

# # Resample all data again and fill so that the count is now included in every row.
# df_ws2         = df_ws2.set_index('datetime').resample('1min').bfill(limit=10)

# # Apply the division on the entire dataframe and clean up.
# df_ws2          = df_ws2.div(df_ws2['count'], axis = 0).reset_index().drop(['group','count'], axis = 1)



# df_ws2['group'] = df_ws2.index 
# df_ws2['count'] = df_ws2.set_index('datetime').resample('1min').bfill(limit=10)['group'].value_counts() #bfill() acts on originally missing data which it should not.
# df_ws2['precip'] = df_ws2['precip']/df_ws2['count']
# df_ws2= df_ws2.set_index('datetime').resample('1min').bfill(limit=10)
# df_ws2= df_ws2.reset_index().drop(['group', 'count'], axis = 1)