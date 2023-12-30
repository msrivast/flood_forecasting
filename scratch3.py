import matplotlib.pyplot as plt


import numpy as np
import pandas as pd

df = pd.read_csv('curated_7_better.csv', usecols=['datetime','precip'])
df['datetime'] = pd.to_datetime(df['datetime'])

# df = df[df["datetime"]>pd.Timestamp("2022-02-17 00:00")]
# df = df[df["datetime"]<pd.Timestamp("2022-02-20 00:00")]

# df = df[df["datetime"]>pd.Timestamp("2022-03-08 12:00")]
# df = df[df["datetime"]<pd.Timestamp("2022-03-11 00:00")]

# df = df[df["datetime"]>pd.Timestamp("2022-05-24 00:00")]
# df = df[df["datetime"]<pd.Timestamp("2022-05-26 00:00")]
# df = df[df["datetime"]>pd.Timestamp("2022-05-26 00:00")]
# df = df[df["datetime"]<pd.Timestamp("2022-05-27 12:00")]

# df = df[df["datetime"]>pd.Timestamp("2022-08-01 00:00")]
# df = df[df["datetime"]>pd.Timestamp("2022-08-11 00:00")]
# df = df[df["datetime"]<pd.Timestamp("2022-08-13 00:00")]

# df = df[df["datetime"]>pd.Timestamp("2022-09-01 04:00")]
# df = df[df["datetime"]<pd.Timestamp("2022-09-06 22:00")]
# df = df[df["datetime"]<pd.Timestamp("2022-09-15 22:00")]

# df = df[df["datetime"]>pd.Timestamp("2022-10-30 04:00")]
# df = df[df["datetime"]<pd.Timestamp("2022-11-01 22:00")]


# df = df[df["datetime"]>pd.Timestamp("2022-11-15 00:00")]
# df = df[df["datetime"]<pd.Timestamp("2022-11-18 00:00")]

df = df.set_index('datetime')

df_out = pd.read_csv('predictions.csv')
# df_out = pd.read_csv('predictions_32N_32N_1900E.csv')
# df_out = pd.read_csv('predictions_32N_4750E.csv')

# df_out = pd.read_csv('predictions_7seqlen_16N_9kE.csv')
# df_out = pd.read_csv('predictions_14N_14kE_BS4.csv')
# df_out = pd.read_csv('predictions _lowlr_1000epochs.csv')
# df_out = pd.read_csv('predictions_10seqlen_5kE_overfit.csv')


df_out['datetime'] = pd.to_datetime(df_out['datetime'])

# df_out['level'] = np.power(10, df_out['level'])
# df_out['Model forecast'] = np.power(10, df_out['Model forecast'])

# print(df_out)

# df_out = df_out[df_out["datetime"]>pd.Timestamp("2022-02-17 00:00")]
# df_out = df_out[df_out["datetime"]<pd.Timestamp("2022-02-20 00:00")]

# df_out = df_out[df_out["datetime"]>pd.Timestamp("2022-03-08 12:00")]
# df_out = df_out[df_out["datetime"]<pd.Timestamp("2022-03-11 00:00")]

# df_out = df_out[df_out["datetime"]>pd.Timestamp("2022-05-24 00:00")]
# df_out = df_out[df_out["datetime"]<pd.Timestamp("2022-05-26 00:00")]
# df_out = df_out[df_out["datetime"]>pd.Timestamp("2022-05-26 00:00")]
# df_out = df_out[df_out["datetime"]<pd.Timestamp("2022-05-27 12:00")]

# df_out = df_out[df_out["datetime"]>pd.Timestamp("2022-08-01 00:00")]
# df_out = df_out[df_out["datetime"]>pd.Timestamp("2022-08-11 00:00")]
# df_out = df_out[df_out["datetime"]<pd.Timestamp("2022-08-13 00:00")]

# df_out = df_out[df_out["datetime"]>pd.Timestamp("2022-09-01 04:00")]
# df_out = df_out[df_out["datetime"]<pd.Timestamp("2022-09-06 22:00")]
# df_out = df_out[df_out["datetime"]<pd.Timestamp("2022-09-15 22:00")]

# df_out = df_out[df_out["datetime"]>pd.Timestamp("2022-10-30 00:00")]
# df_out = df_out[df_out["datetime"]<pd.Timestamp("2022-11-01 00:00")]

# df_out = df_out[df_out["datetime"]>pd.Timestamp("2022-11-15 00:00")]
# df_out = df_out[df_out["datetime"]<pd.Timestamp("2022-11-18 00:00")]

df_out = df_out.set_index('datetime')
plt.plot(850-df_out['level'])
plt.plot(850-df_out['Model forecast'], alpha = 0.7)
plt.title("Water level vs SPIKE prediction(mm)")
plt.legend(["level", "prediction"])
plt.ylabel("mm")
plt.twinx().plot(df['precip'], c='green', alpha = 0.4)
# plt.savefig("training_fit.pdf")
plt.show()