from scipy.interpolate import interp1d
import math
import matplotlib.pyplot as plt 
import numpy as np

BR_list = []
Y_list = []
R_list = []
for i in range(1, 51):
    file_BR = open("data_fixed_sum_300/{}_br.txt".format(i), "r")
    s = file_BR.readline().split(", ")[:-1]
    br = [float(e) for e in s]
    BR_list.append(br)
    file_Y = open("data_fixed_sum_300/{}_Y.txt".format(i), "r")
    s = file_Y.readline().split(", ")[:-1]
    y = [float(e) for e in s]
    Y_list.append(y)
    file_R = open("data_fixed_sum_300/{}_R.txt".format(i), "r")
    s = file_R.readline().split(", ")[:-1]
    r = [float(e) for e in s]
    R_list.append(r)

X = np.linspace(0, 300, 31)

# X = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110, 115, 117, 118, 119, 119.5, 120]
# X = [0, 3, 6, 9, 15, 20, 30, 45, 60, 80, 100, 120, 150, 200, 300]
fig, axes = plt.subplots(nrows=1, ncols=3)
Y_ = np.mean(Y_list, axis=0)
print(Y_)
BR_ = np.mean(BR_list, axis=0)
# fig, axes = plt.subplots(nrows=1, ncols=2)

s1 = [0, 2, 4, 6, 8, 10, 12, 15, 18, 20, 22, 24, 26, 28, 30]
y = axes[0].plot(X[s1], Y_[s1], marker='v', label = r'FTPL')
br = axes[0].plot(X[s1], BR_[s1], marker='v', label=r'SBR')
axes[0].legend()
axes[0].set_xlabel(r'$k_A$')
axes[0].set_ylabel(r'$E(V_A)$')

ftpl_bm = []
sbr_bm = []
n = 32
for i in range(0, len(Y_)):
    y = Y_[i]
    br = BR_[i]
    k_a = X[i]
    k_b = max(X) - k_a
    if X[i] > max(X)/2:
        ftpl_bm.append(y/(n-y) * k_b/k_a)
        sbr_bm.append(br/(n-br) * k_b/k_a)
    else:
        ftpl_bm.append((n-y)/(y) * k_a/k_b)
        sbr_bm.append((n-br)/(br) * k_a/k_b)
print(len(ftpl_bm))
s2 = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
_ = axes[1].plot(X[s2], np.array(ftpl_bm)[s2], marker='v', label="FTPL")
_ = axes[1].plot(X[s2], np.array(sbr_bm)[s2], marker='v', label="SBR")

# axes[1].set_title("Budget Multiplier vs. Budget")
axes[1].legend()
axes[1].set_xlabel(r'$k_A$')
axes[1].set_ylabel('Budget Multiplier')


s3 = [0, 4, 8, 12, 16, 21, 24, 25, 26, 27, 28, 29, 30]

R_ = np.mean(R_list, axis=0)
x_smooth = np.linspace(0, 300, 200)
f = interp1d(X[s3], R_[s3], kind='cubic')
y_smooth = f(x_smooth)
# fig, axes = plt.subplots(nrows=1, ncols=2)
r = axes[2].plot(x_smooth, y_smooth, label = r'Iters')
r = axes[2].scatter(X[s3], R_[s3], marker='v', label = r'Iters')
# axes[2].set_title("Iterations vs. Budget")
axes[2].set_xlabel(r'$k_A$')
axes[2].set_ylabel('FTPL Iterations to Convergence')
fig.tight_layout(pad=0)

