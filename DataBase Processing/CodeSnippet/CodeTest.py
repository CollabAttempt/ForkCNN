import numpy as np

x = np.load('SejongDB Addons.npy')
if 'cap' == x[0]:
    print(x[0])