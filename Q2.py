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
    if k > t_images.size:
        return "invalid k input"
    k_dist = np.array([])
    k_neighbors = np.array([])
    k_labels = np.array([])
    i = 0
    for image in t_images:
        dist = np.linalg.norm(image - q_image)
        if (i==0):
            k_dist = np.insert(k_dist,k_dist.size,dist)
            k_neighbors = image
            k_labels = np.insert(k_labels,k_labels.size,v_labels[i])
        else:
            max_val = np.amax(k_dist)
            if (k_dist.size < k):
                k_dist = np.insert(k_dist,k_dist.size,dist)
                k_neighbors = np.vstack([k_neighbors,image])
                k_labels = np.insert(k_labels,k_labels.size,v_labels[i])
            elif (dist <= max_val):
                max_index = np.where(k_dist == max_val)[0]
                k_dist[max_index] = dist
                k_neighbors[max_index] = image
                k_labels[max_index] = v_labels[i]
        i+=1
    values, counts = np.unique(k_labels, return_counts=True)
    return (values[counts.argmax()])

firs_10_images = train[:10]
firs_10_labels = labels[:10]
image_test = test[0]

print(k_NN(firs_10_images,firs_10_labels,image_test,10))


        










###----Question 2----####