import pandas as pd
import numpy as np
from leastsqdataset import SequenceDataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import  Lasso
from sktime.transformations.series.outlier_detection import HampelFilter



#Read curated file, form a row of the A matrix[precip,precip_past12..............]


# df = pd.read_csv('NHC7_curated_15m_7_past_12_test.csv')#5,1
# df = pd.read_csv('NHC296_curated_15m_7_past_12_test_low_noise.csv')#5,1
df = pd.read_csv('NHC1697_WS3_curated_15m_7_past_12_test.csv')#7,1
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

target = 'level'
# features= ['precip', 'precip_past_12']
features= ['precip']


sequence_length = 28 #7

train_dataset = SequenceDataset(
    df,
    target=target,
    features=features,
    sequence_length=sequence_length,
    test = False
)
print('# not testing data:', len(train_dataset))
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
b = np.array(b)


# NHC7:(window 5(10),sigma 1(2))the smaller the window length, the smaller the sigma needs to be, otherwise we lose peaks if window contains only peaks
# NHC 296: window 10(50), sigma 1(2)
#NHC1697: window 7, sigma 1
# 7,1 works reasonably well for all but its better than 5,1 from NHC1697.
window =7
sigma = 1
transformer = HampelFilter(window_length=window, n_sigma=sigma, k=1.4826, return_bool=True) #k is set for normal distribution
b_hat = transformer.fit_transform(b)
print('Window, Sigma, # outliers:', window, sigma, b_hat.sum())
A_outlier = A[(b_hat.flatten()>0).tolist()]
b_outlier = b[(b_hat.flatten()>0).tolist()]

# reg_nnls = LinearRegression(positive=False)
reg_nnls = Lasso(positive=True) #Performace is WOW!!! works well with False and even without ri stuff though ri makes peak better at 4 hrs!!!

reg_nnls.fit(A_outlier, -b_outlier)

# reg_nnls.fit(A, -b)


w = reg_nnls.coef_
# print(w)

axis = [i for i in range(len(w)-1,-1,-1)]
plt.plot(axis,w)
# plt.xticks(np.arange(27,-1,-1))
plt.xlim(max(axis),min(axis))
plt.xticks(np.arange(0,len(w)-1,1))
plt.show()

# mean_level = mean_level #NHC7
mean_level = 1000 #NHC296
plt.plot(mean_level-b_outlier)
plt.plot(mean_level - -(np.dot(A_outlier,w)+reg_nnls.intercept_))
plt.title('Outliers')
plt.show()

plt.plot(mean_level-b)
plt.plot(mean_level - -(np.dot(A,w)+reg_nnls.intercept_))
plt.title('Not testing set')
plt.show()

b_test = []
A_test = []
for i in range(len(test_dataset)):
    X, y = test_dataset[i]
    b_test += [y]
    A_test += [X.flatten()]

A_test = np.array(A_test)
b_test = np.array(b_test)

# # A_test = np.append(A_test,np.ones([len(A_test),1]),1)

# # plt.plot(mean_level-b_test)
# # plt.plot(mean_level --(A_test@w + reg_nnls.intercept_))
# # plt.show()

ystar_col = 'forecast'

df.loc[(df.valid)&~(df.test), ystar_col] = -(np.dot(A,w) + reg_nnls.intercept_)
df.loc[(df.valid)&(df.test), ystar_col] = -(np.dot(A_test,w) + reg_nnls.intercept_)
# plt.figure(2)
fig, ax1 = plt.subplots()
# print(df)
# df.to_csv('test.csv',float_format='%.0f')

ax1.plot(mean_level - df[(df.valid)&(df.test)][['level','forecast']]) # Test data
# ax1.plot(mean_level - df[(df.valid)&(~df.test)][['level','forecast']]) # Train data

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