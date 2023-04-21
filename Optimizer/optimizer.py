from scipy.interpolate import griddata
import numpy as np
import pandas as pd

pr_lam = 1.64E-08 # price of lambda
pr_put = 5.00E-06 # price of S3 put
pr_get = 4.00E-07 # price of S3 get
D = 50000 # number of training data samples 

# model coefficients
S = float(input("Train coefficient a:"))
a = float(input("Train coefficient a:"))
b = float(input("Train coefficient b:"))
m = float(input("Train coefficient m:"))
k1 = float(input("Trans coefficient k1:"))
k2 = float(input("Trans coefficient k2:"))

# task constraint
E = int(input("Epoch:"))
Tcon = int(input("Time constraint (min):"))
Mlower = int(input("Minimum memory (MB):"))
Mupper = 3008
Mstep = 64
Bg = 1024
gamma = 0.8
Bopt = round(b / (1 / gamma - 1)) # batch size

# upload time and download time with different memory and file size (used to interpolate)
fbw_up = pd.read_excel("bw_up.xls")
fbw_down = pd.read_excel("bw_down.xls")
fm = pd.read_excel("m.xls")
ff = pd.read_excel("f.xls")
# transform the DataFrame into numpy array
fBd = np.array(fbw_down.values).reshape(-1,1)
fBu = np.array(fbw_up.values).reshape(-1,1)
fM = np.array(fm.values).reshape(-1,1)
fF = np.array(ff.values).reshape(-1,1)
points = np.concatenate((fM, fF), axis = 1)

def main():   
    '''
    B: Batch size of aggregators
    W: Num of workers
    M: Memory
    K: Num of aggregators
    A: Batch size of non-aggregators
    T: Total time
    C: Total cost
    '''
    B = Bopt
    W, M, K, A, C, T, flag = Heuristic(B)
    Tscatter = Cal_T(B, W, M, W, A) # total time of ScatterReduce
    Cscatter = Cal_C(B, W, M, W, A) # total cost of ScatterReduce
    acc_rate = (Tscatter - T) / Tscatter * 100 # train speed acceleration rate
    cos_rate = (Cscatter - C)/ Cscatter * 100 # cost reduction rate
    if flag == "Failed": 
        print("No satisfying configuration")
    else:
        print(f"Batch size of aggregators: {B}\nNum of workers: {W}\nMemory: {M}\nNum of aggregators: {K}\nBatch size of non-aggregators: {A}\n\
              Total time: {T:.1f}\nTotal cost: {C:.2}\nTotal time of ScatterReduce: {Tscatter:.1f}\nTotal cost of ScatterReduce: {Cscatter:.2}\n\
              Train speed accelerate by {acc_rate:.1f}%\nCost reduce by {cos_rate:.1f}%")

def Heuristic(B):
    # Pruning-based Heuristic Search Algorithm to Minimize Cost under Time Constraint
    Cmin = float('inf') 
    Wupper = Bg // B
    W = Wupper
    M = Mupper
    A = B
    Tcur = Cal_T(B, Wupper, Mupper, Wupper, A)
    for Wi in range(1, Wupper+1):
        K = Wi
        T = Cal_T(B, Wi, Mupper, K, A)
        if T > Tcon:
            continue
        else:
            Mi = Mlower
            while Mi <= Mupper:
                T = Cal_T(B, Wi, Mi, K, A)
                C = Cal_C(B, Wi, Mi, K, A)
                if C < Cmin and T < Tcon:
                    W = Wi
                    M = Mi
                    Tcur = T
                    Cmin = C
                    break
                Mi += Mstep
    K = W
    if K != 1:
        A = Cal_A(B, W, M, K-1)
        Tnext = Cal_T(B, W, M, K-1, A)
        while Tnext < Tcur:
            K = K - 1
            Tcur = Tnext
            A = Cal_A(B, W, M, K)
            if K == 1:
                break
            Tnext = Cal_T(B, W, M, K-1, A)
        Cmin = Cal_C(B, W, M, K, A)
    A = Cal_A(B, W, M, K)
    if Tcur <= Tcon:
        flag = "Success"
    else:
        flag = "Failed"
    return W, M, K, A, Cmin, Tcur, flag


def Cal_A(B, W, M ,K):
    # decide A based on the aggregation time and training time
    if K == W:
        return B
    Tagg = Cal_Tagg(M, K, W)
    Ttrain = Cal_Ttrain(B, M)
    Amax = round((Tagg + Ttrain) / a * (M + m) - b)  
    Acon = (Bg - B * K) // (W - K)  # constraint from Bg
    return min(Amax, Acon)


def Cal_T(B, W, M, K, A):
    # total time
    Ttrain = Cal_Ttrain(B, M)
    I = Cal_I(B, W, K, A)
    Tcomm = Cal_Tcomm(M, K, W)
    return I * (Tcomm + Ttrain)


def Cal_C(B, W, M, K, A):
    # total cost
    T = Cal_T(B, W, M, K, A)
    C_lam = pr_lam * T * M * W # cost of lambda
    A = Cal_A(B, W, M, K)
    I = Cal_I(B, W, K, A)
    Rup = K * W
    Rdown = 2 * K * (W - 1)
    C_s3 = I * (pr_put * Rup + pr_get * Rdown) # cost of S3
    return C_lam + C_s3


def Cal_Ttrain(B, M):
    # train time in an iteration
    return a * (B + b) / (M + m)


def Cal_Tcomm(M, K, W):
    # commu time in an iteration
    t_up = Cal_Tup(M,K)
    t_down = Cal_Tdown(M,K)
    t_agg = Cal_Tagg(M,K,W)
    Tcomm =  t_up + t_down + t_agg
    return Tcomm 


def Cal_Tup(M, K):
    # uploading time
    t_trans  = Cal_Ttrans(M) # trans time
    fs = S / K # file size
    # interpolate
    up = griddata(points, fBu, [M, fs], method='nearest')[0][0]
    return up * K + t_trans


def Cal_Tagg(M, K, W):
    # aggregation time
    fs = S / K # file size
    # interpolate
    up = griddata(points, fBu, [M, fs], method='nearest')[0][0]
    down = griddata(points, fBd, [M, fs], method='nearest')[0][0]
    t_sync = 0.00119 * W * (fs + 33.7) * (2 - K / W)
    return up + down * (W - 1) +  t_sync
    

def Cal_Tdown(M, K):
    # downloading time
    fs = S / K # file size
    # interpolate
    down = griddata(points, fBd, [M, fs], method='nearest')[0][0]
    if K != 1:
        return down * (K - 1)
    else:
        return down


def Cal_I(B, W, K, A):
    # iteration num
    Ns = D * E
    Bg = B * K + (W - K) * A
    return D*E // Bg


def Cal_Ttrans(M):
    # trans time
    return k1 * M ** k2


if __name__ == '__main__':
    main()


