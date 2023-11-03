from IPython.display import display
import pandas as pd

# data = [[10, 50, 80, 567],
#         [5, 25, 75, 432],
#         [15, 30, 60, 777],
#         [10, 40, 70, 555]]

# df = pd.DataFrame(data,
#                   index=list('ABCD'),
#                   columns=['One', 'Two', 'Three', 'Four'])
# display(df)

# data = { 'Col1': [110, 111, 112, 113],
#         'Col2': [210, 211, 212, 213],
#         'Col3': [310, 311, 312, 313]}

# df = pd.DataFrame(data, index=['p1', 'p2', 'p3', 'p4'])     
# display(df)

#Choosing some data from dataframe
# a = df[0]
# b = df[2]
# c = df[0][0]
# d = df[1][2]
#If we def name of col, we can ref name of col by -> df['columname']['row'] or df.columname[row]

# data = { 'price': [200, 300, 250, 500],
#          'sales': [0, 30, 25, 50],
#          'stock': [8, 0, 5, 1] }

# df = pd.DataFrame(data, index = ['p1', 'p2', 'p3', 'p4'])
# display(df['price'],
#         df['sales'][2],
#         df.stock,
#         df.stock[1],
#         df.stock['p4'])
#In case we need to choose 1 more colums ->df[[]'col1', 'col2', 'col3']]
#or df[start_row : stop_row : step][['col1', 'col2', 'col3'...]]

#iloc and index slicing
# import numpy as np
# import pandas as pd

# data = { 'One' : np.random.randint(100, 1000, 10),
#          'Two' : np.random.uniform(1, 10, 10),
#          'Three' : np.random.rand(10),
#          'Four' : np.random.randint(1, 100, 10),
#          'Five' : np.random.uniform(-10, 10, 10)}

# df = pd.DataFrame(data)
# display(df[['One', 'Three', 'Five']],
#         df[:][['Two', 'Three', 'Four']],
#         df.iloc[2:5, 3:5],
#         df.iloc[::, [0, 2, 4]])

# import pandas as pd

# data = { 'price': [200, 300, 250, 500],
#          'sales': [0, 30, 25, 50],
#          'stock': [8, 0, 5, 1]}
# df = pd.DataFrame(data)
# display(df[df['price'] >= 300],
#         df[(df.price >= 200) & (df.price <= 300)],
#         df[df['stock'] == 0],
#         df[['price', 'stock']][df.stock > 0])


#Some function in DataFrame
#count(), sum(), min(), max(), mean(), describe()-> Show important statistics
#info()-> show data about dataframe, head(n)-> show first row (if not input n will show just fifth order)
#tail(n)-> like head but show last row, rename(), replace(), drop(columns=['col1', 'col2'])

# import numpy as np
# import pandas as pd

# data = { 'One': np.random.randint(100, 1000, 10),
#          'Two': np.random.uniform(1, 10, 10),
#          'Three': np.random.rand(10),
#          'Four': np.random.randint(1, 100, 10),
#          'Five': np.random.uniform(-10, 10, 10)}
# df = pd.DataFrame(data)
# display(
#     df.sum(),
#     df.sum()['One'],
#     df.max(),
#     df.max()['One'],
#     df.describe()
# )

# df.info()

# df.rename(columns={'One':'Col1', 'Two':'Col2'})

# df.replace({10: 11, 20: 22, 30: 33}, inplace=True) #inplace=True -> save result in object
# df.drop(columns=['One', 'Four'], inplace=True)


# import numpy as np
# import pandas as pd

# data = { 'One': np.random.randint(100, 1000, 10),
#          'Two': np.random.uniform(1, 10, 10),
#          'Three': np.random.rand(10),
#          'Four': np.random.randint(1, 100, 10),
#          'Five': np.random.uniform(-10, 10, 10)}

# df = pd.DataFrame(data)
# with pd.option_context('display.max_rows', 6): display(df) #couple of first and last rows work together

