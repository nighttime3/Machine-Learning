# import matplotlib.pyplot as plt
# import numpy as np

# y = [5, 20, 10, 15, 10]
# x = ['Mon', 'Tue', 'Wed',
#      'Thu', 'Fri']
# plt.plot(x,y)
# plt.show()

# y = np.random.randint(1, 11, 5)
# x = list('ABCDE')
# plt.rcParams['figure.figsize'] = (10, 5) # width=10", height=5"
# #plt.figure(figsize=(10,5))

# plt.plot(x,y)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn')

# y = (15, 12, 19, 13, 16, 17, 12)
# x = ('Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat')
# plt.title('Weekly Sales', color='blue', alpha=0.8)
# plt.ylabel('Amount', color='m', fontsize=14)
# plt.xlabel('Weekday', color='g', fontsize=14)
# plt.yticks(color='r')
# plt.xticks(rotation=45, color='r', fontsize=13) #rotation='vertical'/'horizontal or put degree
# plt.plot(x,y)
# plt.show()

#when we need to change value on x ticks
# y = [5, 15, 15, 20, 10]
# x = [1, 3, 5, 7, 9]
# plt.plot(y)
# plt.xticks([0, 1, 2, 3, 4])
# plt.show()

#if we need just some data, use axis for def gap between data we need 
# y = np.random.randint(10, 100, 11)
# x = np.arange(2010, 2021)
# plt.axis([2015, 2020, 10, 100])

# plt.plot(x,y)
# plt.show()

#decoration graph
# y = [6, 8, 5, 7]
# x = [1, 2, 3, 4]  
# plt.plot(x, y, '--ro')
# plt.grid(color='g', alpha=0.1, lw=2, ls='--')
# plt.show()

#plot graph more than 1 
# y = [5, 15, 15, 20, 10]
# x = [1, 2, 3, 4, 5]
# plt.plot(x, y, ls='-', c='r')

# y = [15, 20, 10, 5, 15]
# plt.plot(x, y, ls='--', c='g')

# y = [10, 5, 15, 15, 20]
# plt.plot(x, y, ls=':', c='b')
# plt.show()

#input detail for each line with plt.legend
# y = [10, 15, 15, 20, 10]
# x = [1, 2, 3, 4, 5]
# plt.plot(x, y, ls='-', c='r', label='Product #1')

# y = [15, 25, 10, 15, 20]
# plt.plot(x, y, ls='--', c='g', label='Product #2')

# y = [10, 20, 15, 10, 15]
# plt.plot(x, y, ls=':', c='b', label='Product #3')

# plt.legend(loc='best')
# plt.show()

#Scatter 

y = np.random.randint(1, 101, 20)
x = np.arange(1, 21)
plt.scatter(x, y, color='g', marker='o', s=80)
plt.yticks(np.arange(0, 101, 10))
plt.xticks(x)
plt.show()

