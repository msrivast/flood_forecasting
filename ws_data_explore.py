import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# array = np.loadtxt('oconee_2022.txt',delimiter=',', skiprows = 2143, usecols=[2, 3])

# Show time on the x-axis

df_airport = pd.read_csv('oconee_2022.txt', usecols=['valid', 'precip_in'])
df_airport['valid'] = pd.to_datetime(df_airport['valid'])
df_airport = df_airport[df_airport["valid"]<pd.Timestamp("2022-12-07 18:00")]
# print(df.valid.dtype)
# print(df.head())
# plt.plot(df_airport['valid'], df_airport['precip_in']*25.4)

df_airport = df_airport.set_index("valid")
plt.plot(df_airport*25.4,':')
# plt.title("Airport")
# plt.ylabel("mm")
# plt.savefig("airport.png")
# plt.show()


df_ws2 = pd.read_csv('WS2_final_2022.csv', usecols=['datetime', 'precip'])
df_ws2['datetime'] = pd.to_datetime(df_ws2['datetime'])
# print(df_ws2.head())
# Accumulate the rainfall over the past hour and report one entry at the top of the hour
# df = df_ws2[df_ws2["datetime"]<pd.Timestamp("2022-01-01 02:00")]
df_ws2['group'] = df_ws2.index
df_ws2['count'] = df_ws2.set_index('datetime').resample('1min').bfill(limit=7)['group'].value_counts() # bfill() writes 0 for no data -> 7/22 - 8/03 - > manually deleted data between 7/22 to 7/28 
df_ws2          = df_ws2.set_index('datetime').resample('1min').bfill(limit=7)
df_ws2          = df_ws2.div(df_ws2['count'], axis = 0).reset_index().drop(['group','count'], axis = 1)
df_ws2 = df_ws2.set_index("datetime")
df_ws2 = df_ws2.resample('1H', label = 'right', closed = 'right').sum(min_count=1)
# df_ws2 = df_ws2.resample('1H', label = 'right', closed = 'right').sum(min_count=1) # writes 0 for no data -> 7/28 - 8/03(fixed by min_count)
# df_ws2.to_csv("ws2.csv")
plt.plot(df_ws2, alpha = 0.7)
# plt.title("WS #2")
# plt.ylabel("mm")
# plt.legend(["Airport", "WS#2"])
# plt.show()


df_ws3 = pd.read_csv('WS3_final_2022.csv', usecols=['datetime', 'precip'])
df_ws3['datetime'] = pd.to_datetime(df_ws3['datetime'])
# print(df_ws3.head())
# Accumulate the rainfall over the past hour and report one entry at the top of the hour
# df = df_ws3[df_ws3["datetime"]<pd.Timestamp("2022-04-01 02:00")]
df_ws3['group'] = df_ws3.index
df_ws3['count'] = df_ws3.set_index('datetime').resample('1min').bfill(limit=7)['group'].value_counts()
df_ws3          = df_ws3.set_index('datetime').resample('1min').bfill(limit=7)
df_ws3          = df_ws3.div(df_ws3['count'], axis = 0).reset_index().drop(['group','count'], axis = 1)
df_ws3 = df_ws3.set_index("datetime")
df_ws3 = df_ws3.resample('1H', label = 'right', closed = 'right').sum(min_count=1) # writes 0 for no data -> 10/13 - 10/19(fixed by min_count) and no data before march
# df_ws3.to_csv("ws3.csv")
plt.plot(df_ws3, alpha = 0.7)
# plt.title("WS #3")
# plt.ylabel("mm")
# plt.legend(["Airport", "WS#3"])
# plt.show()

plt.title("Weather Stations")
plt.ylabel("mm")
plt.legend(["Airport", "WS#2", "WS#3"])
plt.savefig("all.pdf")
plt.show()

# print(array)
# plt.plot(array*25.4)
# plt.show()


# import matplotlib
# gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
# non_gui_backends = matplotlib.rcsetup.non_interactive_bk
# print ("Non Gui backends are:", non_gui_backends)
# print ("Gui backends I will test for", gui_env)
# for gui in gui_env:
#     print ("testing", gui)
#     try:
#         matplotlib.use(gui,warn=False, force=True)
#         from matplotlib import pyplot as plt
#         print ("    ",gui, "Is Available")
#         plt.plot([1.5,2.0,2.5])
#         fig = plt.gcf()
#         fig.suptitle(gui)
#         plt.show()
#         print ("Using ..... ",matplotlib.get_backend())
#     except:
#         print ("    ",gui, "Not found")