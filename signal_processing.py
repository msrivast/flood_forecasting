import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

df_ws2 = pd.read_csv('WS3_final_2022.csv', usecols=['datetime', 'precip'])
df_ws2['datetime'] = pd.to_datetime(df_ws2['datetime'])

# df_ws2 = df_ws2[df_ws2["datetime"]>pd.Timestamp("2022-5-24 18:00")]
# df_ws2 = df_ws2[df_ws2["datetime"]<pd.Timestamp("2022-5-28 00:00")] 

df_ws2 = df_ws2.set_index("datetime")
print((df_ws2<0.1).sum()- (df_ws2==0).sum())
# plt.plot(df_ws2[(df_ws2<0.1) & (df_ws2>0)])
plt.plot(df_ws2[(df_ws2<0.1)])
print((df_ws2[df_ws2<0.1]).sum())
plt.show()
# df_ws2[df_ws2<0.1] = 0
# df_ws2 = df_ws2.reset_index()

# print(df_ws2)