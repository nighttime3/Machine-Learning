#Numpy
#Building array
# import numpy as np
# a = np.array([2, 4, 6]) #array from list
# b = np.array(range(1, 11)) #array from function range()
# c = np.array([7, 11, 101, 555]) #array from tuple
# d = np.array([3 * x for x in range(1, 6)])

#function in numpy
# a = np.zeros(5) #[0 0 0 0 0]
# b = np.ones(5) #[1 1 1 1 1]
# c = np.full(5, 8) #[8 8 8 8 8]
# d = np.arange(1, 6) #[1 2 3 4 5]
# e = np.arange(1, 10, 2) #[1 3 5 7 9]
# f = np.linspace(1, 10, 4) #[1 4 7 10] #linspace build values between 1-10 for 4 values by each space x-y have the same values
# g = np.linspace(1, 10, 6) #[1 2.8 4.6 6.4 8.2 10]
# h = np.random.uniform(1, 101, 10) #random 1-100 for 10 numbers
# i = np.random.rand(10)

# a = np.array([7, 11, 108, 1009, 101, 555])
# print(a[0])
# print(a[1] + a[3])
# print(a[-1])
# print(a[-5])
# print(*a)
# for value in a: print(value)

#index slicing like a list[start:stop:step]
# import numpy as np
# arr = np.array([2, 7, 11, 101, 108, 360, 404, 555, 747, 1009, 2020])
# print(
#     arr[5:],
#     arr[3:9],
#     arr[1:10:2])
#Fancy indexing
# arr = np.array([2, 7, 11, 101, 108, 360, 404, 555, 747])
# idx = [0, 2, 4, 8]
# a = arr[idx]
# print(a)

# a2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(a2d)

# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(a, a[0, 0],a[1, 2], a[2, 2], sep='\n\n')

# arr = np.array([7, 11, 108, 1009, 101, 555])
# arr2 = arr+10

# a = np.array([2, 7, 11, 101])
# b = np.array([10, 8, 9, 99])
# a + b #[12 15 20 200] -> a and b must have 4 elements for cal

# arr = np.random.randint(10, 101, 10)
# a = arr[arr >= 50]
# b = arr[arr % 2 == 0]
# c = arr[(arr >= 20) & (arr<=60)]
# d = arr[(arr != 33) | (arr != 51) | (arr != 89)]

#function numpy
#sum(),min(),max(),unique(),reshape(),mean()
# a1 = np.arange(1, 11)
# ar = a1.reshape(2, 5) #or ar = np.arange(1, 11).reshape([2,5])

# a2 = np.random.randint(1, 10, (2,6))
# b = a2.reshape(3, 4)
# print(a2)
# print(b)  #when we dont know rows or cols we use -1 instead in (-1,2) & (3,-1).