import numpy as np
import matplotlib.pyplot as plt
N = 200000
n = 20
A_assist = np.random.binomial(size=N*n, n=1, p= 0.5)
A = A_assist.reshape((N,n))
epsilon_L = np.linspace(0,1,50)
x = []         #epsilonvalues
y = []         #emprical_prob

def emprical_prob_cal(empir_mean: float,epsilon: float) -> int:
    res = 0
    abs_cal = abs(empir_mean - 0.5)
    if(abs_cal > epsilon):
        res = 1
    return res

def empirical_mean_cal(mat : np.array) -> list:
    L = [(sum(mat[i])/n) for i in range(0,N)]
    return(L)

L = empirical_mean_cal(A)
print(L)
for epsilon in epsilon_L:
    x.append(epsilon)
    cnt = 0
    for e_mean in L:
        cnt += emprical_prob_cal(e_mean,epsilon)
    y.append(cnt/N)

x_val = np.array(x)
y_val = np.array(y)
plt.title("Q1.b")
plt.xlabel("Epsilon")
plt.ylabel("Empirical Probability")
plt.plot(x_val, y_val, color ="blue")
plt.show()


