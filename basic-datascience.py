#range()
# range(stop)
# range(start, stop)
# range(start, stop, step)

# one_to_ten = range(1, 11)
# print(*one_to_ten) # print(*variable) type or show data respectively, like using for (loop)

# even_numbers = range(0, 20, 2)
# print(*even_numbers)

# count_down = range(10, 0, -1)
# print(*count_down)

# print(*one_to_ten)
# for x in one_to_ten: print(x, sep=' ', end=' ')


#Exameple list

# a = list([1, 2, 3, 4, 5])
# b = list(range(3, 6))
# c = list[1, True, 'Three', 4.5]
# d = list('abcd')
# e = [1, 2, 3, 4, 5]
# f = [range(3, 6)]
# g = [1, True, 'Three', 4.5]

# data = [1, 2, 3, 4, 5]
# print(data[0])
# print(data[3])
# print(*data)
# for x in data: print(x, sep=' ')


#some operator can use with list

# a = [1, 3, 5]
# b = [2, 4, 6, 8]
# c = a + b
# d = b + a #list a + b and list b + a not the same result !

# e = [1, 3, 5, 0]
# print(e * 2)  #[1, 3, 5, 0, 1, 3, 5, 0]

# f = [0, 1, 2, 3, 4, 5, 6, 7]
# print(f[1:3])
# print(f[4:10]) # if data is not enough will count just we have
# print(f[3:])
# print(f[:5])
# print(f[1:6:2])

#Built-in use with list = len(list), max,min(list), sum(list), enumerate(list), del(list)
# a = [1, 3, 5, 7, 9]
# print(len(a))
# print(sum(a))

# b = [108, 1009, 101, 7, 11]
# print(max(b))
# print(min(b))

# c = [108, 1009, 7 , 11]
# for (i, v) in enumerate(c):
#     print(f'{i} - {v}')

# nums = [1, 2, 4, 5]
# del nums[2]

#Two-Dimensional List
# nums = [[0, 1, 2 ,3], [4, 5, 6, 7], [8,9]]
# print(nums[0])
# print(nums[1])
# print(len(nums))
# print(len(nums[0]))
# print(len(nums[2]))

# colors = [['red', 'green', 'blue'], ['black', 'white'], ['yellow']]
# print(colors[2])
# print(len(colors))
# print(len(colors[1]))

# print(colors[0][0])
# print(colors[0][1])
# print(colors[1][0])
# print(colors[1][1])

# colors = [['red', 'green', 'blue'], ['black', 'white']]
# for cols in colors:
#     for c in cols:
#         print(c, end=' ')

# colors[1].append('yellow')
# colors[1] += ['pink', 'purple']
# print(colors[1])

# if 'green' in colors[0]: print('yes')

# if cols in colors:
#     if 'white' in cols: print('yes')


#Tuple
# a = [1, 2, 3]
# list = [4, 5, 6]
# b = tuple(list) #create tuple from list
# c = tuple([7, 8, 9])
# d = (1,) #if we have only 1 element -> place , after first element

#Dictionary
# countires = {'th':'Thailand', 'jp':'Japan', 'kr':'Korea'}
# info = {'name':'Beer', 'age':28, 'height':165, 'single':False}
# ranges = {'a':range(80, 100), 'b':range(70, 9), 'c':range(60, 9)}
# odd_evens = {'odd':[1, 3, 5, 7, 9], 'even':[2,4,6,8]}

# print(countires['th'])
# print(countires['jp'])

# a = odd_evens['odd'][0] #place [] for call each index in tuple
# b = odd_evens['even'][3]

# for key in countires:
#     print(countires[key])
 
#keys() -> reading all keys sent data in type of tuple
#values() -> reading all values (tuple)
#items() -> tuple(keys,values)

# countries = {'th':'Thailand', 'jp':'Japan', 'kr':'Korea'}
# for key in countries.keys(): # or key in countries:
#     print(countries[key])

# values = countries.values()
# for v in values: print(v, ' ', end=' ')

# for (k, v) in countries.items(): print(f'{k}: {v}')


