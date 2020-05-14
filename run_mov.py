import json
from classes import Election, Candidate
from optimize_mov import *
import random

random.seed(410)
def run(i):
    with open ('json/comm' + str(i) + '.json', 'r') as fp:
        data = json.load(fp)

    opinion_attr = "sex"

    # X = [0, 3, 6, 9, 15, 20, 30, 45, 60, 80, 100, 120, 150, 200, 300]
    X = np.linspace(0, 120, 13)
    n = 32
    Y = [] # FTPL payoff
    BR = [] # BR payoff
    file_BR = open("data_fixed_sum_slurm/{}_br.txt".format(i), "w")
    file_Y = open("data_fixed_sum_slurm/{}_Y.txt".format(i), "w")
    A = Candidate("A", 0, 1, n)
    B = Candidate("B", 0, 0, n)
    e = Election(data, n, [A, B], 10, opinion_attr, rand=False)
    for x in X:
        print("BUDGET ", x)
        A.k=x
        B.k = 100-x
        e.update_network()

        ftpl(e, 5, .1, x)

        ftpl_mean = e.calculate_mean()
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

    file_BR.close()
    file_Y.close()
    return X, np.array(Y)-np.array(BR)

def main():
    for i in range(1, 3):
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

main()
