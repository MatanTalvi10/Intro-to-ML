import numpy as np
import matplotlib.pyplot as plt
N = 200000
n = 20
def emprical_prob_cal(empir_mean: float,epsilon: float) -> int:
    res = 0
    abs_cal = abs(empir_mean - 0.5)
    if(abs_cal > epsilon):
        res = 1
    return res

def empirical_mean_cal(mat : np.array) -> list:
    L = [(sum(mat[i])/n) for i in range(0,N)]
    return(L)