import json
from classes import Election, Candidate
from optimize_mov import *
import random
import os
import sys

random.seed(410)
def run(i):
    with open ('json/comm' + str(i) + '.json', 'r') as fp:
        data = json.load(fp)

    opinion_attr = "sex"

    # X = [0, 5, 10]

    # X = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110, 115, 117, 118, 119, 119.5, 120]
    # X = [0, 3, 6, 9, 15, 20, 30, 45, 60, 80, 100, 120, 150, 200, 300]
    X = np.linspace(0, 300, 31)
    n = 16
    Y = [] # FTPL payoff
    BR = [] # BR payoff
    R = [] # iters
    file_BR = open("data_sum300_n16/{}_br.txt".format(i), "w")
    file_Y = open("data_sum300_n16/{}_Y.txt".format(i), "w")
    file_R = open("data_sum300_n16/{}_R.txt".format(i), "w")

    blockPrint()
    A = Candidate("A", 0, 1, n)
    B = Candidate("B", 0, 0, n)
    e = Election(data, n, [A, B], 10, opinion_attr, rand=False)

    for x in X:
        print("BUDGET ", x)
        A.k=x
        B.k = max(X)-x
        print("BUDGETS 1", A.k, B.k)

        e.update_network()

        ftpl_mean, r = ftpl(e, .05,  x)
        R.append(r)
        file_R.write(str(r)+ ', ')
        print("BUDGETS 2", A.k, B.k)
        # print("RUN XA", sum(A.X), A.X)
        # print("RUN XB", sum(B.X), B.X)
        print("FTPL A", A.ftpl_history)
        print("FTPL B", B.ftpl_history)

        Y.append(ftpl_mean)
        file_Y.write(str(ftpl_mean)+ ', ')

        A.X = mov_oracle(e, A, np.zeros(n))
        B.X = mov_oracle(e, B, np.zeros(n))
        br_mean = e.calculate_mean()
        BR.append(br_mean)

        file_BR.write(str(br_mean) + ', ')
        for file in [file_BR, file_Y]:
            file.flush()
        print("means: {}, {}".format(ftpl_mean, br_mean))
    enablePrint()
    print("Network", i)
    print("BR", BR)
    print("Y", Y)
    print("R", R)
    file_BR.close()
    file_Y.close()
    file_R.close()
    return X, np.array(Y)-np.array(BR)

def main():
    for i in range(10, 40):
        run(i)
    # Xs, Ys = [], []
    # X, Y = run(3)
    # Xs.append(X)
    # Ys.append(Y)
    # X_mean = np.mean(Xs, axis=0)
    # Y_mean = np.mean(Ys, axis=0)
    # print(X_mean, Y_mean)
    # plt.scatter(X, Y)
    # plt.show()
    # plt.savefig('ftpl.png', dpi=300)

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
main()

