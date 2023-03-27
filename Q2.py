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

firs_1000_images = train[:1000]
firs_1000_labels = labels[:1000]

def test_algo(n: int,k: int,) -> float:
    images_t = train[:n]
    labels_t = labels[:n]
    j = 0
    correct = 0
    for t_image in test:
        res = k_NN(images_t,labels_t,t_image,k)
        wanted = labels_t[j]
        if (str(res) == wanted):
            correct+=1
        j+=1
    return (correct/test.size)

print(test_algo(1000,10))



        










###----Question 2----####