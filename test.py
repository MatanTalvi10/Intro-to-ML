import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.random
from sklearn.datasets import fetch_openml


my_array = np.array([1, 2.2, 4, 4, 4, 5, 6, 7, 12, 12, 12])
print(np.unique(my_array, return_counts=True))
values, counts = np.unique(my_array, return_counts=True)
print(values[counts.argmax()])
print(str(9) == '9')