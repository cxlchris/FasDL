import numpy as np
from scipy import optimize

def train_func(B, M, p):
    """
    train time function: T = a(B+b)/(M+m)
    p is the parameters a, b, m
    """
    a, b, m = p
    return a * (B + b) / (M + m)


def trans_func(M, p):
    """
    trans time function: T = k1 * M ** k2
    p is the parameters k1, k2
    """
    k1, k2 = p
    return k1 * M ** k2


def train_residuals(p, T, B, M):
    """
    get the diff of real T and func
    """
    return T - train_func(B, M, p)

def trans_residuals(p, T, M):
    """
    get the diff of real T and func
    """
    return T - trans_func(M, p)

def main():
    # load profiling data
    B = np.loadtxt('profiling.txt', usecols=0) # batchsize
    M = np.loadtxt('profiling.txt', usecols=1) # memory
    T_train = np.loadtxt('profiling.txt', usecols=2) # train time
    T_trans = np.loadtxt('profiling.txt', usecols=3) # trans time

    # fit coefficients with least square
    train_plsq = optimize.leastsq(train_residuals, np.array([0, 0, 0]), args=(T_train, B, M))
    trans_plsq = optimize.leastsq(trans_residuals, np.array([0, 0]), args=(T_trans, M))

    a, b, m = train_plsq[0]
    print(f"train coefficients:\na = {a:.3}\nb = {b:.3}\n m = {m:.3}")
    k1, k2 = trans_plsq[0]
    print(f"trans coefficients:\nk1 = {k1:.3}\n k2 = {k2:.3}")

