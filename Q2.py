###----Question 2----####
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.random
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]
print(data[0].shape)

def k_NN(t_images: np.array,v_labels: np.array,q_image: np.array,k: int) -> np.array:
    k_dist = numpy.array([])
    k_neighbors = np.zeros(k)
    k_labels = np.zeros(k)
    for image in t_images:
        i = 0
        dist = np.linalg.norm(image - q_image)
        if (i==0):
            k_dist = np.insert(k_dist,k_dist.size,dist)
            k_neighbors = np.insert(k_neighbors,k_neighbors.size,image)
            k_labels = np.insert(k_labels,k_labels.size,v_labels[i])
        else:
            max_val = np.amax(k_dist)
            if (k_dist.size < k):
                k_dist = np.insert(k_dist,k_dist.size,dist)
                k_neighbors = np.insert(k_neighbors,k_neighbors.size,image)
                k_labels = np.insert(k_labels,k_labels.size,v_labels[i])
            elif (dist <= max_val):
                max_index = np.where(k_dist == max_val)[0]
                k_dist[max_index] = dist
                k_neighbors[max_index] = image
                k_labels[max_index] = v_labels[i]
        i+=1
    pred_lab = np.bincount(k_labels).argmax()
    return pred_lab


        










###----Question 2----####