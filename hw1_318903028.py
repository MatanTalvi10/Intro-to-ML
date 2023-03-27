import numpy as np
import numpy.random
import math
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
data = mnist['data']
labels = mnist['target']
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


# ---a--- #
def k_NN(t_images: np.array,v_labels: np.array,q_image: np.array,k: int) -> np.array:
    if k > t_images.size:
        return "invalid k input"
    distlabel = []  
    LD = {}
    for i in range(len(t_images)):
        dist = np.linalg.norm(t_images[i] - q_image)
        labelIdx = v_labels[i]
        distlabel.append((dist, labelIdx))
    distlabelsorted = sorted(distlabel, key=lambda item: item[0])
    distlabelsorted = distlabelsorted[:k]
    for tuple_assist in distlabelsorted:
        if tuple_assist[1] in LD:
            LD[tuple_assist[1]] += 1
        else:
            LD[tuple_assist[1]] = 1
    return max(LD, key=LD.get)

# ---b--- #
def test_algo(n: int,k: int,) -> float:
    n_images = train[:n]
    n_labels = train_labels[:n]
    labelsRes = []
    for t in range(len(test)):
        labelsRes.append(k_NN(n_images,n_labels,test[t],k))     
    wrong = 0
    for j in range(len(test_labels)):
        if test_labels[j] != labelsRes[j]:
            wrong += 1
    accuracy = 1 - wrong/len(test_labels)
    return (accuracy)

# ---c--- #
k_val = [i for i in range(1,101)]
acc_val = [None]*100
for k in range(1,101):
    res = test_algo(1000,k)
    acc_val[k-1] = res
x_val = np.array(k_val)
y_val = np.array(acc_val)
plt.title("Q2.c")
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.plot(x_val, y_val, color ="blue")
plt.show()

# ---d--- #
n_val = [i for i in range(100, 5001, 100)]
acc_val = [None]*(len(n_val))
for i in range(len(n_val)):
    res = test_algo(n_val[i],1)
    acc_val[i] = res
x_val = np.array(n_val)
y_val = np.array(acc_val)
plt.title("Q2.d")
plt.xlabel("n values")
plt.ylabel("accuracy")
plt.plot(x_val, y_val, color ="blue")
plt.show()


