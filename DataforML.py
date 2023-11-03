from IPython.display import display
from sklearn.datasets import make_classification
import pandas as pd

# data = [[10, 20, 'A'],
#         [5, 15, 'B'],
#         [9, 13, 'B'],
#         [8, 16, 'A']]

# df = pd.DataFrame(data, columns=['x1', 'x2', 'y'])
# display(df)

#build datasets with make_classification-> simple and a lot of data
# features, target = make_classification(n_samples=10, n_features=4, n_classes=2, random_state=0)
# print('Features:\n', features)
# print('Target:\n', target)

# x, y = make_classification(n_samples=100, n_features=6, n_informative=3,
#                            n_redundant=2, n_repeated=0, n_clusters_per_class=2,
#                            n_classes=3, random_state=0)

# #if we need DataFrame
# df = pd.DataFrame(x, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6'])
# df['y'] = y

# with pd.option_context('display.max_rows', 6): display(df)

#create sample data with make_regression-> The results are continuous numbers.
# from sklearn.datasets import make_regression
# import pandas as pd
# x, y = make_regression(n_samples=100, n_features=5, random_state=0)

# df = pd.DataFrame(x, columns=['x1', 'x2', 'x3', 'x4', 'x5'])
# df['y'] = y

# with pd.option_context('display.max_rows', 10): display(df)
#This regression is easy and simple more than classification


#build dataset with make_blobs -> Cluster data
# from sklearn.datasets import make_blobs
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# x, y = make_blobs(n_samples=100, centers=3, random_state=1)

# df = pd.DataFrame(x, columns=['x1', 'x2'])
# df['y'] = y
# with pd.option_context('display.max_rows', 6): display(df)

# colors = np.array(['#999', '000', '#ddd'])
# plt.scatter(df['x1'], df['x2'], color=colors[df['y']])
# plt.show()

#make_circles ->circle model ex. Support Vector Machines
# from sklearn.datasets import make_circles
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# x, y = make_circles(n_samples=100, noise=0.08, random_state=0)

# df = pd.DataFrame(x, columns=['x1', 'x2'])
# df['y'] = y
# with pd.option_context('display.max_rows', 6): display(df)

# colors = np.array(['#999', '000'])
# plt.scatter(df['x1'], df['x2'], color=colors[df['y']])
# plt.show()

#How to check missing value
#isnull().any()-> check NaN in colums
#isnull().sum()-> count NaN each colums
#dropna()-> del row have 'NaN' 
#fillna()-> change NaN to any(input)

import pandas as pd
import numpy as np

# df = pd.read_csv('D:/Machine Learning/Data science/test-missing-values.csv')
# display(df)
# print(df.isnull().any())
# print()
# print(df.isnull().sum())

# df['volume'].isnull.any() #when we need check just some colums
# df[['voluime', 'weight']].isnull().sum()

#dropna()
# df = pd.read_csv('D:/Machine Learning/Data science/test-missing-values.csv')
# df.dropna(inplace=True)
# display(df)

#fillna()-> most using (mean), (method='ffill') or (method='pad'), (method='bfill') or bfill()
# df = pd.read_csv('D:/Machine Learning/Data science/test-missing-values.csv')
# v_mean = df['volume'].mean()
# df['volume'].fillna(v_mean, inplace=True)
# df['weight'].fillna(df['weight'].mean(), inplace=True)
# display(df)

#pad and bfill
# df = pd.read_csv('D:/Machine Learning/Data science/test-missing-values.csv')
# df['volume'].fillna(method='pad', inplace=True)
# df['weight'].fillna(method='bfill', inplace=True)
# display(df)

#Encoding
#first method
# from sklearn.preprocessing import LabelEncoder

# encoder = LabelEncoder()

# data = ['Apple', 'Mango', 'Grape', 'Durian', 'Mango', 'Grape']
# encoder.fit(data)

# enc_data1 = encoder.transform(data)
# print(enc_data1)
# #[0 3 2 1 3 2]

# enc_data2 = encoder.transform(['Mango', 'Durian', 'Durian'])
# print(enc_data2)
# #[3 1 1]

# enc_data3 = encoder.transform(['Grape', 'Mango', 'Mango', 'Durian'])
# print(enc_data3)
# #[2 3 3 1]

# enc_data4 = encoder.transform(['mango', 'Mangosteen', 'Mango', 'Melon'])
# print(enc_data4)
#Error Because mango isn't Mango and Mangosteen and Melon not found in fit()

#Encoding second method
# from sklearn.preprocessing import LabelEncoder

# encoder = LabelEncoder()

# data = ['Apple', 'Mango', 'Grape', 'Durian', 'Mango', 'Grape']
# encoder = LabelEncoder()

# enc_data1 = encoder.fit_transform(data)
# print(enc_data1)

# enc_data2 = encoder.transform(['Mango', 'Durian', 'Durian'])
# print(enc_data2)

# enc_data2 = encoder.transform(['Apple', 'Grape', 'Durian', 'Grape'])
# print(enc_data2)

#Decoder 
# from sklearn.preprocessing import LabelEncoder

# encoder = LabelEncoder()

# data = ['Apple', 'Mango', 'Grape', 'Durian', 'Mango', 'Grape']
# encoder = LabelEncoder()

# enc_data = encoder.fit_transform(data)
# print(enc_data)

# inv = encoder.inverse_transform(enc_data)
# print(inv)

# print(encoder.inverse_transform([1, 1, 0, 2]))

#DataFrame can encoder just 1 column, if we need to encoder more 1 column 
#we have to build seperate encoder
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv('D:/Machine Learning/Data science/test-label-encoder.xlsx')
# display(df)
# enc_weather = LabelEncoder()
# enc_weather.fit(df['Weather'])
# df['Weather'] = enc_weather.transform(df['Weather'])

# enc_timeofweek = LabelEncoder().fit(df['TimeOfWeek'])
# df['TimeOfWeek'] = enc_timeofweek.transform(df['TimeOfWeek'])

# df['TimeOfDay'] = LabelEncoder().fit_transform(df['TimeOfDay'])

# display(df)


#We can encoder multiple columns in DataFrame,Simple method to do is apply() and def fit_transform()
# df = pd.read_csv('D:/Machine Learning/Data science/test-label-encoder.xlsx')
# display(df)

# encoder = LabelEncoder()
# df = df.apply(encoder.fit_transform)
# print(df)

#if we need to encoder just one columns
# encoder = LabelEncoder()
# col = ['Weather']
# df[col] = df[col].apply(encoder.fit_transform)
# display(df)

#if we need to encoder multiple columns
# encoder = LabelEncoder()
# cols = ['Weather', 'TimeOfWeek', 'TimeOfDay']
# df[cols] = df[cols].apply(encoder.fit_transform)
# display(df)

#encoder index 0-1
# encoder = LabelEncoder()
# df.iloc[:, 0:2] = df.iloc[:, 0:2].apply(encoder.fit_transform)
# display(df)

#Scaling data
# from sklearn.preprocessing import StandardScaler

# data = [[2, 2000],
#         [3.5, 5500],
#         [1.75, 2150],
#         [3.14, 4130],
#         [2.25, 2564]]

# scaler = StandardScaler()
# scaler.fit(data)
# data2 = scaler.transform(data)
# print(data2)
# print()

# data_test = [[1.23, 3210]]
# print(scaler.transform(data_test))

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

scaler = StandardScaler()

df = pd.read_csv('D:/Machine Learning/Data science/test-standard-scaler.csv')


#fit() ... and transform()
# scaler.fit(df[['volume', 'weight']])
# df[['volume', 'weight']] = scaler.transform(df[['volume', 'weight']])
# display(df)

# fit_transform()
# df[['volume', 'weight']] = scaler.fit_transform(df[['volume', 'weight']])
# display(df)

#If we need to input more data when we're scaling data, form need to be list or 2D array
# data = [[1.19, 1000]]
# print(scaler.transform(data))
# print(scaler.transform([[2.3, 1234]]))

