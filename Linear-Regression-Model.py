#y = mx + c
#m = y2-y1/x2-x1
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# data = [[3, 10],
#         [5, 7],
#         [6, 6],
#         [7, 4.5],
#         [9, 3.25]]

# arr = np.array(data)
# x = arr[:, 0]
# y = arr[:, 1]

# plt.figure(figsize=(4, 3))
# plt.scatter(x, y, s=50)
# plt.show()

#Correlation

# data = [[3, 10], [5, 7], [6, 6], [7, 4.5], [9, 3.25]]
# df = pd.DataFrame(data, columns=['x', 'y'])
# display(df.corr().round(2))
# If values approaches 1.00 or -1.00 meaing we can predict data through Linear Regression


#prediction with scikit-learn

from sklearn.linear_model import LinearRegression
# model = LinearRegression()

#create list or 2D array
#1st Method
# data = [[2, 6],
#         [4, 10],
#         [6, 11],
#         [10, 22],
#         [12, 25]]
# data = np.array(data)
# #change feature 2D array to 1 column with reshape()
# x = data[:, 0].reshape(-1, 1)
# #change result to 2D with reshape (do or dont, whatever)
# # y = data[:, 1].reshape(-1, 1)
# y = data[:, 1]
# print(x)
# print(y)
# print(type(data))


#2nd Method
#Def rows in feature/target to 2D
# x = [[2], [4], [6], [10], [12]]
# y = [[6], [10], [11], [22], [25]]
#y = [6, 10, 11, 22, 25]
# print(x)
# print(y)

#3rd Method
#def feature to 1D first, Then do to 2D with reshape()
# x = [2, 4, 6, 8, 10, 12]
# x = np.array(x).reshape(-1, 1)
# y = [[6], [10], [11], [22], [25]] 
# y = np.array(y).reshape(-1, 1)
# print(x,y)    

#In case we lead data from DataFrame
# df = pd.read_csd(...)
# x = df['col1']
# x = np.array(x).reshape(-1, 1)
# y = df['label']

#Train Model
# model.fit(x, y)
# y_5 = model.predict([[5]])
# y_9 = model.predict([[9]])
# print(y_5)
# print(y_5[0])
# print(y_9)
# print(y_9[0])

# x_predict = [[5], [9], [11]]
# y_predict = model.predict(x_predict)
# print(y_predict)
# print('x = 5, y =', y_predict[0])
# print('x = 9, y =', y_predict[1])
# print('x = 11, y =', y_predict[2])

# intercept = model.intercept_
# print(intercept)

# coefficient = model.coef_
# print(print(coefficient[0]))

# x = [2, 4, 6, 8, 10, 12]
# x = np.array(x).reshape(-1, 1)

# y = [6, 10, 11, 14, 22, 25]

# model = LinearRegression()
# model.fit(x, y)

# y_5 = model.predict([[5]])
# print('x = 5, y=', y_5[0])

# y_9 = model.predict([[9]])
# print('x = 9, y=', y_9[0])

# y_11 = model.predict([[11]])
# print('x = 11, y=', y_11[0])

# print('intercept =', model.intercept_)
# print('coefficient =', model.coef_[0])
# Linear Regression equation y = B0(intercept) + B1(coefficient)x + E(Error)


import matplotlib.pyplot as plt

# x = list(range(5, 61, 5))
# y = [8, 42, 102, 113, 148, 206, 202, 230, 360, 354, 427, 432]
# plt.scatter(x, y)
# plt.show()



# x = list(range(5, 61, 5))
# y = [8, 42, 102, 113, 148, 206, 202, 230, 360, 354, 427, 432]
# plt.scatter(x, y)
# plt.show()

# x = np.array(x).reshape(-1, 1)

# model = LinearRegression()
# model.fit(x, y)

# x_predict = [[8], [18], [33], [59]]
# y_predict = model.predict(x_predict)

# #result for 2 points float 
# for (i, p) in enumerate(x_predict):
#     cal = '[:.2f]'.format(y_predict[i])
#     print(f'Exercise for {p[0]} minutes burn {cal} calories')
# print()
# ic = '{:.2f}'.format(model.intercept_)
# ce = '{:.2f}'.format(model.coef_[0])
# print(f'Equation predict: y = {ic} + ({ce})x')

# df = pd.read_csv('D:/Machine Learning/Data science/umbrellas_sold_1.xlsx')
# # with pd.option_context('display.max_rows', 6): display(df)

# x = df['rainfall_mm']
# y = df['umbrellas_sold']
# plt.scatter(x, y)
# plt.show()

# x = np.array(x).reshape(-1,1)
# y = df['umbrellas_sold']
# model = LinearRegression()
# model.fit(x, y)

# ic = '{:,.2f}'.format(model.intercept_)
# ce = '{:,.2f}'.format(model.coef_[0])
# print()
# x_predict = [[90], [100], [120]] #The amount of rain you have to predicted.
# y_predict = model.predict(x_predict)

# for (i, p) in enumerate(x_predict):
#     sale = '{:.0f}'.format(y_predict[i])
#     t = f'amount of rain {p[0]} mm'
#     t += f'umbrellas sold {sale} ea'
#     print(t)



# Model Evaluation
"""
How to cal accuracy -> R^2 = 1- RSS^2/TSS^2 
(R^2 = Accuracy[value in 0-1 if 0.85 bring this value *100 = 85% Acc])
RSS = Difference between real results and predicted results
TSS = Difference betweem real results and mean values real results
"""
#when we use this table to cal RSS^2 and TSS^2
# x = [2, 4, 6, 8, 10, 12]
# x = np.array(x).reshape(-1, 1)

# y = [6, 10, 11, 14, 22, 25]

# model = LinearRegression()
# model.fit(x, y)

# y_5 = model.predict([[5]])
# print('x = 5, y=', y_5[0])

# y_9 = model.predict([[9]])
# print('x = 9, y=', y_9[0])

# y_11 = model.predict([[11]])
# print('x = 11, y=', y_11[0])

# print('intercept =', model.intercept_)
# print('coefficient =', model.coef_[0])

"""
intercept = 1.26 
coefficient = 1.91
f(xi) = intercept + coef(xi)
y' = (6+10+11+14+22+25)/6 = 14.68
f(x[0]) = 1.26 (1.91 * 2) = 5.08 ->[0] = 2, [1] = 4, [3] = 6
RSS^2 = (6 - 5.08)^2 = 0.84
TSS^2 = (6 - 14.68)^2 = 75.34
loop all
R^2 = 1 - (13.6/270.51) = 0.94
Accuracy = 94% -> High Acc is mean we can use this data for decision
"""

#Calculate Accuracy with scikit-learn
from sklearn.metrics import r2_score

# x = [2, 4, 6, 8, 10, 12]
# y = [6, 10, 11, 14, 22, 25]
# x = np.array(x).reshape(-1, 1)

# model = LinearRegression()
# model.fit(x, y)

# score = model.score(x, y)

#y_predict = model.predict(x)
#score = r2_score(y,y_predict)

# print('R-Squared:', score)
# accuracy = score * 100
# accuracy = '{:,.2f}'.format(accuracy)
# print(f'Accuracy: {accuracy}%')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# df = pd.read_csv('D:/Machine Learning/Data science/umbrellas_sold_1.xlsx')
# x = df['rainfall_mm']
# x = np.array(x).reshape(-1, 1)
# y = df['umbrellas_sold']

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# model = LinearRegression()
# model.fit(x_train, y_train)

# score = model.score(x_test, y_test)

# print('R-Squared:', "{:.4f}".format(score))
# print()

# accuracy = score*100
# accuracy = '{:,.2f}'.format(accuracy)
# print(f'Accuracy: {accuracy}%')


#RMSE -> Error of accuracy
"""
Mean Square Error(MSE) = (yi - f(xi)^2)/n
Root Mean Square Error = (RMSE)^0.5
RMSE = (13.6/6)^0.5 = 1.50
y_predict difference from y_true = +/- 1.50
Reliable data should have a low RMSE and high accuracy.
"""
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
# call csv for checking type of linear regression (positive), This data can use for prediction with Linear regression
# df = pd.read_csv('D:/Machine Learning/Data science/birth_weight.xlsx')
# x = df['Gestational_Age_wks']
# y = df['Birth_Weight_gm']
# plt.scatter(x, y)
# plt.show()


#Prediction
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.metrics import r2_score

# df = pd.read_csv('D:/Machine Learning/Data science/birth_weight.xlsx')
# with pd.option_context('display.max_rows', 6): display(df)

# x = ['Gestational_Age_wks']
# x = df[x]
# y = np.array(x).reshape(-1, 1)
# y = df['Birth_Weight_gm']

# model = LinearRegression()
# model.fit(x, y)
# y_predict = model.predict(x)

# mse = mean_squared_error(y, y_predict)
# rmse = math.sqrt(mse)
# print('MSE:', '{:,.2f}'.format(mse))
# print('RSME:', '{:,.2f}'.format(rmse))

# score = model.score(x, y)
# # score = r2_score(y, y_predict)
# print('R-squared:', '{:,.2f}'.format(score))

# ic = '{:.2f}'.format(model.intercept_)
# ce = '{:.2f}'.format(model.coef_[0])
# print(f'prediction equation: y = {ic} + ({ce})x')
# # Accuracy = 76%

#Multiple Linear Regression -> x1,x2,y
# from sklearn.linear_model import LinearRegression
# import numpy as np

# d = [[8, 4, 45],
#      [7, 5, 44],
#      [8, 6, 50],
#      [6, 6, 43],
#      [9, 5, 45],
#      [8, 3, 44],
#      [9, 4, 40],
#      [6, 5, 43]]

# d = np.array(d)
# x = d[:, 0:2]
# y = d[:, 2]

# model = LinearRegression()
# model.fit(x, y)

# ic = '{:,.2f}'.format(model.intercept_)
# ce1 = '{:,.2f}'.format(model.coef_[0])
# ce2 = '{:,.2f}'.format(model.coef_[1])

# print(f'Prediction equation: y = {ic} + ({ce1})x1 + ({ce2}x2)')
"""
y = B0 + B0x1 + B2x2 + B3x3 + ... + BkXk
B0 = y_intercept
B1,B2,B3 = Coefficient
x1,x2,x3 = Feature
"""

# x_predict = [[6, 6], [7, 6]]
# y_predict = model.predict(x_predict)
# p_6_6 = "{:,.2f}".format(y_predict[0])
# p_7_6 = "{:,.2f}".format(y_predict[1])
# print((f'x1 = 6, x2 = 6 y = {p_6_6}'))
# print((f'x1 = 6, x2 = 6 y = {p_7_6}'))

import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('D:/Machine Learning/Data science/stock_index_price.csv')
# x1 = df['interest_rate']
# x2 = df['unemployment_rate']
# y = df['stock_index_price']

# #graph x1,y
# plt.figure(figsize=(4, 3))
# plt.scatter(x1, y, c='b')
# plt.xlabel('interest_rate')
# plt.ylabel('stock_index_price')
# plt.show()

# #graph x2,y
# plt.figure(figsize=(4, 3))
# plt.scatter(x2, y, c='r')
# plt.xlabel('interest_rate')
# plt.ylabel('stock_index_price')
# plt.show()

#checking from corr()
# display(df.corr().round(2))
"""
interest_rate               1.00              -0.97               0.96
unemployment_rate          -0.97               1.00              -0.99
stock_index_price           0.96              -0.99               1.00
From data, This is a positive and negative relative.
So we can use this data to train model with Multiple Linear Regression
"""

# df = pd.read_csv('D:/Machine Learning/Data science/stock_index_price.csv')
# with pd.option_context('display.max_rows', 6): display(df)

# x = df[['interest_rate', 'unemployment_rate']]
# y = df['stock_index_price']

# model = LinearRegression()
# model.fit(x, y)

# #predict stock_index_price when interest_rate = 2 and umployment_rate = 5
# #predict stock_index_price when interest_rate = 2.2 and umployment_rate = 5.7
# x_predict = [[2, 5], [2.2, 5.7]]
# y_predict = model.predict((x_predict))
# for (i, x_p) in enumerate(x_predict):
#     p = '{:,.2f}'.format(y_predict[i])
#     print(f'interest_rate: {x_p[0]}, unemployment_rate: {x_p[1]}, stock_index_price: {p}')
# y_predict_all = model.predict(x)
# sc = r2_score(y, y_predict_all)
# print(f'R-Squared:', '{:,.2f}'.format(sc))


import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv('D:/Machine Learning/Data science/co2_emission.csv')
# with pd.option_context('display.max_rows', 6): display(df)

# x1 = df['volume']
# x2 = df['weight']
# y = df['co2']

# plt.scatter(x1, y, c='b')
# plt.xlabel('volume')
# plt.ylabel('co2')
# plt.show()

# plt.scatter(x2, y, c='r')
# plt.xlabel('volume')
# plt.ylabel('co2')
# plt.show()


#Cal acc
# x = df[['volume', 'weight']]
# y = df['co2']

# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# model = LinearRegression()
# model.fit(x_train, y_train) #get data in model

# #predict co2_emission volume = 2300 and weight = 1300
# x_predict = [[2300, 1300]]
# y_predict = model.predict((x_predict))

# for (i, x_p) in enumerate(x_predict): 
#     co2 = '{:,.2f}'.format(y_predict[i])
#     print(f'volume: {x_p[0]}, weight: {x_p[1]}, co2_emission: {co2}')

# score = model.score(x_test, y_test)
# print('R-squared:', '{:,.2f}'.format(score))

# ic = '{:,.2f}'.format(model.intercept_)
# ce1 = '{:,.2f}'.format(model.coef_[0])
# ce2 = '{:,.2f}'.format(model.coef_[1])
# eq = f'{ic} + ({ce1})x1 + ({ce2})x2'
# print(f'Prediction Equation: y = {eq}')

"""
In case low R-Squared -> The data are not related.
sometimes model is not linear regression but still have position/negative relation
So we can use Polynomial Regression instead
y = B0 + B1x + B2x^2 + B3x^3 + ... + Bkx^k
Watch graph first then desicion we should to use Polynomial Regression or not.
choose degree for each graph but careful 'Overfitting'
"""

# import matplotlib.pyplot as plt

# x = [0, 20, 40, 60 ,80 , 100]
# y = [0.0002, 0.0012, 0.0060, 0.0300, 0.0900, 0.2700]

# plt.scatter(x, y)
# plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import numpy as pd
from sklearn.metrics import r2_score

# x = [0, 20, 40, 60 ,80 , 100]
# y = [0.0002, 0.0012, 0.0060, 0.0300, 0.0900, 0.2700]
# # plt.scatter(x, y, c='b')
# # plt.show()

# x = np.array(x).reshape(-1, 1)
# pf = PolynomialFeatures(degree=3) #try degree = 2,3,4
# x_poly = pf.fit_transform(x)

# model = LinearRegression()
# model.fit(x_poly, y)

# #prediction when temp = 50, 70 and 95
# x_predict = [[50], [70], [95]]
# y_predict = model.predict(pf.fit_transform(x_predict))

# for (i, x_p) in enumerate(x_predict):
#     pressure = '{:,.4f}'.format(y_predict[i])
#     print(f'Temperature: {x_p[0]}, Pressure: {pressure}')

# y_predict2 = model.predict(x_poly)
# plt.scatter(x, y, c='b')
# plt.plot(x, y_predict2, color='r')
# plt.show()


#ph_soid and avg_plant_growth
# x = [6.0, 6.2, 6.4, 6.6, 6.8, 7.0, 7.2]
# y = [25.4, 33.0, 50.8, 53.3, 53.3, 30.5, 22.9] #cm

# plt.figure(figsize=(4, 3))
# plt.scatter(x, y)
# plt.show()

# x = np.array(x).reshape(-1, 1)

# pf = PolynomialFeatures(degree=4)
# x_poly = pf.fit_transform(x)

# model = LinearRegression()
# model.fit(x_poly, y)

# x_predict = [[6.5], [6.7], [6.9]]
# y_predict = model.predict(pf.transform(x_predict))
# for (i, x_p) in enumerate(x_predict):
#     g = '{:.4f}'.format(y_predict[i])
#     print(f'pH = {x_p[0]}, Average Plant Growth(cm) = {g}')

# y_predict_all = model.predict(x_poly)
# score = r2_score(y, y_predict_all)
# print('R-Squared =', '{:,.3f}'.format(score))

# y_predict2 = model.predict(x_poly)

# plt.figure(figsize=(4, 3))
# plt.scatter(x, y, c='b')
# plt.plot(x, y_predict2, c='r')
# plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('D:/Machine Learning/Data science/electricity-consumption.csv')
with pd.option_context('display.max_rows', 6): display(df)

x = df['home_size']
y = df['kilowatt_hours_per_month']

pf = PolynomialFeatures(degree=2)
x = np.array(x).reshape(-1, 1)
x_poly = pf.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

x_predict = [[1500], [2000]]
y_predict = model.predict(pf.transform(x_predict))

for (i, x_p) in enumerate(x_predict):
    kw = '{:,.2f}'.format(y_predict[i])
    print(f'home_size: {x_p[0]}, kw per hours: {kw}')

score = model.score(x_poly, y)
print('R-Squared:', '{:,.3f}'.format(score))

y_predict_all = model.predict(x_poly)
plt.scatter(x, y, c='b')
plt.plot(x, y_predict_all, c='r')
plt.show()
