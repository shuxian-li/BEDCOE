import math
import numpy as np
import warnings


# silence the warning
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')


def Gmean_compute(recall):
    Gmean = 1
    n = 0
    for r in recall:
        if math.isnan(r):
            n = n + 1
        else:
            Gmean = Gmean * r
    Gmean = pow(Gmean, 1/(len(recall)-n))
    return Gmean


def confusion_online(cf, y_t, p_t, theta=0.99):
    cf[int(y_t), int(p_t)] = theta * cf[int(y_t), int(p_t)] + 1 - theta

    return cf


def accuracy(cf):
    correct = np.sum(np.diag(cf))
    total = np.sum(cf)
    acc = correct / total

    return acc


def pf_online(S, N, y_t, p_t, theta=0.99):
    c = int(y_t)  # class 0 or 1
    S[c] = (y_t == p_t) + theta * (S[c])
    N[c] = 1 + theta * N[c]

    recall = S / N
    gmean = Gmean_compute(recall)
    return recall, gmean, S, N


if __name__ == '__main__':
    theta = 1
    y = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
    p = [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    S = np.zeros([2])
    N = np.zeros([2])
    for t in range(len(y)):
        [recall, Gmean, S, N] = pf_online(S, N, y[t], p[t], theta=0.99)
        print(recall)
        print(Gmean)