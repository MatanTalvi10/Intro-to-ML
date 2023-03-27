import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.random
from sklearn.datasets import fetch_openml

k_dist = np.full((1, ), np.inf)
k_dist = np.insert(k_dist,k_dist.size,2)
k_dist = np.insert(k_dist,k_dist.size,8)
print(k_dist)
print(k_dist[1])
print(np.amax(k_dist))
