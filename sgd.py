#################################
# Your name: Matan Talvi
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt
from scipy import special


"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    # TODO: Implement me

    data_len = data.shape[0]
    w = np.zeros((data.shape[1]))
    for t in range(1, T+1):
        i = np.random.randint(0, data_len, 1)[0]
        eta_t = eta_0 / t
        y = labels[i]
        x = data[i]
        if y * w @ x < 1:
            w = (1 - eta_t) * w + eta_t * C * y * x
        else:
            w = (1 - eta_t) * w
    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    # TODO: Implement me
    data_len = data.shape[0]
    w = np.zeros((data.shape[1]))
    for t in range(1, T+1):
        i = np.random.randint(0, data_len, 1)[0]
        eta_t = eta_0 / t
        y = labels[i]
        x = data[i]
        exp = (-y * x) * special.softmax([-y * np.inner(w, x)])[0]
        w = w - (eta_t * exp)
    return w

#################################

# Place for additional code

def Q1A(train_data, train_labels):
    T=1000
    C= 1
    s_len=10
    eta_options= [10**i for i in range(-5, 4)]
    acc= np.zeros(len(eta_options))
    i=0
    for eta_0 in eta_options:
        for n in range(s_len):
            w= SGD_hinge(train_data, train_labels, C, eta_0, T)
            acc[i]+= for_accuracy(train_data, train_labels, w)
        acc[i]= acc[i]/s_len
        i+=1
    plt.title("accuracy - SGD_Hinge loss per eta_0")
    plt.xlabel('eta_0')
    plt.ylabel('average acc')
    plt.xscale('log')
    plt.plot(eta_options, acc, marker='o')
    plt.show()
    return eta_options[np.argmax(acc)]

def Q1B(train_data, train_labels, eta_0):
    C_for_b= [10**i for i in range(-5, 4)]
    T=1000
    s_len = 8
    acc = np.zeros(len(C_for_b))
    i = 0
    for C in C_for_b:
        for n in range(s_len):
            w = SGD_hinge(train_data, train_labels, C, eta_0, T)
            acc[i] += for_accuracy(train_data, train_labels, w)
        acc[i] = acc[i] / s_len
        i+=1

    plt.title("acc of SGD_Hinge loss per C")
    plt.xlabel('C')
    plt.ylabel('average acc')
    plt.xscale('log')
    plt.plot(C_for_b, acc, marker='o')
    plt.show()

    return C_for_b[np.argmax(acc)]

def Q1C(train_data, train_labels, eta_0, C):
    T=20000
    w = SGD_hinge(train_data, train_labels, C, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()

def Q1D(train_data, train_labels, eta_0, C):
    T = 20000
    w = SGD_hinge(train_data, train_labels, C, eta_0, T)
    return for_accuracy(train_data, train_labels, w)


def Q2A(train_data, train_labels):
    T = 1000
    eta_options = [10 ** i for i in range(-7, 5 + 1)]
    s_len = 10
    acc = np.zeros(len(eta_options))
    i = 0
    for eta0 in eta_options:
        for n in range(s_len):
            w = SGD_log(train_data, train_labels, eta0, T)
            acc[i] += for_accuracy(train_data, train_labels, w)
        acc[i] = acc[i] / s_len
        i+=1

    plt.title("accuracy of SGD_Log loss per eta_0")
    plt.xlabel('eta_0')
    plt.ylabel('average acc')
    plt.xscale('log')
    plt.plot(eta_options, acc, marker='o')
    plt.show()

    return eta_options[np.argmax(acc)]

def Q2B(train_data, train_labels, eta_0):
    T = 20000
    w = SGD_log(train_data, train_labels, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.show()
    return for_accuracy(train_data, train_labels, w)

def Q2C(X_train, Y_train, eta_0):
    """
    Train the classifier with T=20000 and found in sections a
    Plot the norm of w as a function of the iteration
    """
    T = 20000
    w_norms = norm_iterations(X_train, Y_train, eta_0, T)
    iterates = np.arange(1, 20001)
    plt.plot(iterates, w_norms)
    plt.xlabel('iteration')
    plt.ylabel('norm(w)')
    plt.show()

def norm_iterations(data, labels, eta_0, T):
    """
    Calculate w norm for each of T iterates
    """
    w = np.zeros(784)  
    w_norms = np.zeros(T+1)
    for t in range(1, T + 1):
        i = np.random.randint(data.shape[0]) 
        y = labels[i]
        x = data[i]
        exp = (-y * x) * special.softmax([-y * np.dot(w, x)])[0]
        eta = eta_0 / t
        w = w - (eta * exp)
        w_norms[t] = np.linalg.norm(w)
    return w_norms[1:]

def for_accuracy(data, labels, w):
    data_len = data.shape[0]
    count_correct = 0
    for i in range(data_len):
        y = labels[i]
        x = data[i]
        if (x @ w >= 0 and y==1)  or (x @ w < 0 and y==-1):
            count_correct += 1
    return count_correct / data_len

main()
#################################