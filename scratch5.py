# GOAL: Add 24hr total to curated hourly data
import pandas as pd

# Read file 
df = pd.read_csv('curated_7_better.csv')
SEQ_LENGTH = 7

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

df['last_24'] = df['precip'].rolling(window='24H').sum()

# df.to_csv('test.csv')



def get_item(i):
    # create a list of SEQ_LENGTH length comprising of lists of features ending at index i
    l = [list(df.drop("level",axis=1).iloc[i])]
    for m in range(1,SEQ_LENGTH):
        if (df.index[i] - df.index[i-m] != pd.Timedelta(m,'h')):
            # l = [0, df['precip'][i-m-24:i-m].last('24H').sum()] + l
            l = [[0, df['precip'][df.index[i] - pd.Timedelta(23+m,'h'): df.index[i] - pd.Timedelta(m,'h')].sum()]] + l
        else:
            l = [list(df.drop("level",axis=1).iloc[i-m])] + l
    return l

print(get_item(0))
