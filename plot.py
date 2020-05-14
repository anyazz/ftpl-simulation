import math
import matplotlib.pyplot as plt 
import numpy as np

BR_list = []
Y_list = []
for i in range(2, 15):
    file_BR = open("data/{}_br.txt".format(i), "r")
    s = file_BR.readline().split(", ")[:-1]
    br = [float(e) for e in s]
    BR_list.append(br)
    file_Y = open("data/{}_Y.txt".format(i), "r")
    s = file_Y.readline().split(", ")[:-1]
    y = [float(e) for e in s]
    Y_list.append(y)

X = [0, 3, 6, 9, 15, 20, 30, 45, 60, 80, 100, 120, 150, 200, 300]
fig, axes = plt.subplots(nrows=1, ncols=2)
Y_ = np.mean(Y_list, axis=0)
BR_ = np.mean(BR_list, axis=0)
# fig, axes = plt.subplots(nrows=1, ncols=2)
y = axes[0].scatter(X, Y_, label = r'FTPL')
br = axes[0].scatter(X, BR_, label=r'SBR')
axes[0].set_title("Expected Votes for A vs. Budget")
axes[0].legend()
axes[0].set_xlabel(r'$k_A$')
axes[0].set_ylabel(r'$E(V_A)$')

ftpl_bm = []
sbr_bm = []
for i in range(1, len(Y_)):
    y = Y_[i]
    br = BR_[i]
    if X[i] > 0:
        ftpl_bm.append(y/(32-y) * 40/(X[i]))
        sbr_bm.append(br/(32-br) * 40/(X[i]))
    else:
        ftpl_bm.append((32-y)/(y) * X[i]/40)
        sbr_bm.append((32-br)/(br) * X[i]/40)
print(ftpl_bm)
print(sbr_bm)
_ = axes[1].scatter(X[1:], ftpl_bm, label="FTPL")
_ = axes[1].scatter(X[1:], sbr_bm, label="SBR")

axes[1].set_title("Budget Multiplier vs. Budget")
axes[1].legend()
axes[1].set_xlabel(r'$k_A$')
axes[1].set_ylabel('Budget Multiplier')

plt.show()
