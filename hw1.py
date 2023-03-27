import numpy as np
import matplotlib.pyplot as plt
from functions import *
import math
###----Question 1----####
A_assist = np.random.binomial(size=N*n, n=1, p= 0.5)
A = A_assist.reshape((N,n))
epsilon_L = np.linspace(0,1,50)
x = []         #epsilonvalues
y = []         #emprical_prob
z = [2*(math.exp(-2*n*(epsilon**2))) for epsilon in epsilon_L]         #Hoeffding bound
L = empirical_mean_cal(A)

for epsilon in epsilon_L:
    x.append(epsilon)
    cnt = 0
    for e_mean in L:
        cnt += emprical_prob_cal(e_mean,epsilon)
    y.append(cnt/N)

x_val = np.array(x)
y_val = np.array(y)
z_val = np.array(z)
plt.title("Q1.b")
plt.xlabel("Epsilon")
plt.ylabel("Empirical Probability")
plt.plot(x_val, y_val, color ="blue")
plt.show()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y, 'g-')
ax2.plot(x, z, 'b-')
ax1.set_xlabel('Epsilon')
ax1.set_ylabel('Empirical Probability', color='g')
ax2.set_ylabel('Hoeffding bound', color='b')
plt.title("Q1.c")
plt.show()

###----Question 1----####
