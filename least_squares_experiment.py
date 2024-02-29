import pandas as pd
import numpy as np
from leastsqdataset import SequenceDataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import  Ridge
from sklearn.linear_model import  Lasso
from sklearn.linear_model import  ElasticNet



#Read curated file, form a row of the A matrix[precip,precip_past12..............]


df = pd.read_csv('NHC7_curated_15m_7_past_12_test.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

target = 'level'
features= ['precip', 'precip_past_12']
# features= ['precip']

# df[(df.valid)&(~df.test)]['level'].hist(bins=10)
# plt.show()
df.loc[(df.valid)&(~df.test),'binned'] = np.searchsorted(np.histogram_bin_edges(df[(df.valid)&(~df.test)]['level'].values,bins=100), df[(df.valid)&(~df.test)]['level'].values)
df.loc[(df.valid)&(~df.test),'bin_size'] = df[(df.valid)&(~df.test)].groupby(df[(df.valid)&(~df.test)].binned)['binned'].transform('count')



sequence_length = 28 #7

train_dataset = SequenceDataset(
    df,
    target=target,
    features=features,
    sequence_length=sequence_length,
    test = False
)
print(len(train_dataset))
test_dataset = SequenceDataset(
    df,
    target=target,
    features=features,
    sequence_length=sequence_length,
    test = True
)
# i = 0
# X, y = train_dataset[i]
# print(X)
# print(y)

A = []
b = []
for i in range(len(train_dataset)):
# for i in range(2):
    X, y = train_dataset[i]
    b += [y]
    A += [X.flatten()]

A = np.array(A)
# b = np.log(np.array(b))
# b_max = max(b)
# b = np.exp(1-b/b_max)
b = np.array(b)

#Relative importance
frequency = df[(df.valid)&(~df.test)]['bin_size'].values
# ri = 1/frequency
# ri = np.exp(-frequency)
ri = -np.log((frequency)/max(frequency))
# ri = 1/np.log(frequency+1)
# ri=(max(b)-b)/np.log(frequency+0.1)
# ri = np.ones(len(frequency));

# A = np.append(A,np.ones([len(A),1]),1)
# print(type(A))

# transformer = HampelFilter(window_length=10, n_sigma=3, k=1.4826, return_bool=True)
# b_hat = transformer.fit_transform(b)

# reg_nnls = LinearRegression(positive=True)
# reg_nnls = Ridge(positive=True)
reg_nnls = Lasso(positive=True) #Performace is WOW!!! works well with False and even without ri stuff though ri makes peak better at 4 hrs!!!
# reg_nnls = ElasticNet(positive=True)

# reg_nnls.fit(A, -b)
reg_nnls.fit(A, -b, sample_weight=ri)
# reg_nnls.fit(A, -b, sample_weight=1/b)
# reg_nnls.fit(A, -b, sample_weight=1/(b-min(b)+1))
# reg_nnls.fit(A, -b, sample_weight=max(b)-b+0.1)

w = reg_nnls.coef_
# print(w)

axis = [i for i in range(len(w)-1,-1,-1)]
plt.plot(axis,w)
# plt.xticks(np.arange(27,-1,-1))
plt.xlim(max(axis),min(axis))
plt.xticks(np.arange(0,len(w)-1,1))
plt.show()


plt.plot(850-b)
plt.plot(850 - -(np.dot(A,w)+reg_nnls.intercept_))
plt.show()

b_test = []
A_test = []
for i in range(len(test_dataset)):
    X, y = test_dataset[i]
    b_test += [y]
    A_test += [X.flatten()]

A_test = np.array(A_test)
# b_test = np.log(np.array(b_test))
# b_test = np.exp(b_max-b_test)
b_test = np.array(b_test)

# A_test = np.append(A_test,np.ones([len(A_test),1]),1)

# plt.plot(850-b_test)
# plt.plot(850 --(A_test@w + reg_nnls.intercept_))
# plt.show()

ystar_col = 'forecast'

# df.loc[(df.valid)&~(df.test), ystar_col] = np.exp(-(np.dot(A,w) + reg_nnls.intercept_))
# df.loc[(df.valid)&(df.test), ystar_col] = np.exp(-(np.dot(A_test,w) + reg_nnls.intercept_))
# df.loc[(df.valid)&~(df.test), ystar_col] = (1-np.log(np.dot(A,w) + reg_nnls.intercept_))*b_max
# df.loc[(df.valid)&(df.test), ystar_col] = (1-np.log(np.dot(A_test,w) + reg_nnls.intercept_))*b_max
df.loc[(df.valid)&~(df.test), ystar_col] = -(np.dot(A,w) + reg_nnls.intercept_)
df.loc[(df.valid)&(df.test), ystar_col] = -(np.dot(A_test,w) + reg_nnls.intercept_)
# plt.figure(2)
fig, ax1 = plt.subplots()
# print(df)
df.to_csv('test.csv',float_format='%.0f')

# ax1.plot(850 - df[(df.valid)&(df.test)][['level','forecast']]) # Test data
ax1.plot(850 - df[(df.valid)&(~df.test)][['level','forecast']]) # Train data

plt.legend(["level", "prediction"],loc=4)
plt.ylabel("Stream height change (mm)")
plt.xticks(rotation=45)
# ax2=ax1.twiny()
# plt.twinx().plot(df[df.valid]['precip'], c='lightsteelblue', alpha = 0.9)
plt.twinx().plot(df['precip'], c='lightsteelblue', alpha = 0.9)
plt.gca().invert_yaxis()
plt.ylabel("Rainfall (mm)")
fig.tight_layout()
plt.show()
# print('Train R2: ', reg_nnls.score(A,-b))
# print('Test  R2: ', reg_nnls.score(A_test,-b_test))