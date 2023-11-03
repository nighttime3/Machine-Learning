#Training set -> 75% / Testing set -> 25%
from IPython.display import display
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
# x, y = make_classification(n_samples=10, n_features=4,
#                            n_classes=2, random_state=0)

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# print(x_train)
# print(y_train)
# print()
# print(x_test)
# print(y_test)



# df = pd.read_csv('D:/Machine Learning/Data science/test-standard-scaler.csv')
# x = df.iloc[:, 2:4] #df[['volume', 'weight']]
# y = df['co2']

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# display(x_train)
# display(y_train)
# print()
# display(x_test)
# display(y_test)

#when random_state changed
df = pd.read_csv('D:/Machine Learning/Data science/test-standard-scaler.csv')
# x = df.iloc[:, 2:4] #df[['volume', 'weight']]
# y = df['co2']

# x_train, _, _, _ = train_test_split(x, y, random_state=0)
# display(x_train)
# print('random_state = 0')

# x_train, _, _, _ = train_test_split(x, y, random_state=10)
# display(x_train)
# print('random_state = 0')

# x_train, _, _, _ = train_test_split(x, y, random_state=100)
# display(x_train)
# print('random_state = 0')

#We should to do train_test_split() first, Then do scaling
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)
# display(x_train)
# display(x_test)



