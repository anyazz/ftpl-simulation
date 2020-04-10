import json
from classes import Election, Candidate
from optimize_mov import *
import random

random.seed(410)
def run(i):
    with open ('comm' + str(i) + '.json', 'r') as fp:
        data = json.load(fp)

    opinion_attr = "sex"

    X = np.concatenate([np.linspace(0,  80, 17)])
    n=32
    Y = [] # FTPL payoff
    BR = [] # BR payoff
    file_X = open("FTPL_X.txt", "w")
    file_BR = open("FTPL_BR.txt", "w")
    file_Y = open("FTPL_Y.txt", "w")
    file_XA = open("FTPL_XA.txt", "w")
    file_XB = open("FTPL_XB.txt", "w")
    XAs, XBs = [], []
    A = Candidate("A", 0, 1, n)
    B = Candidate("B", 40, 0, n)
    e = Election(data, n, [A, B], 10, opinion_attr, rand=False)
    for x in X:
        print("BUDGET ", x)
        file_X.write(str(x) + ', ')
        A.k=x
        e.update_network()

        ftpl(e, 5, .1, x)

        ftpl_mean = e.calculate_mean()
        XAs.append(A.X)
        file_XA.write(str(A.X) + ',')
        XBs.append(B.X)
        file_XB.write(str(B.X) + ',')
        Y.append(ftpl_mean)
        file_Y.write(str(ftpl_mean)+ ', ')

        A.X = mov_oracle(e, A, np.zeros(n))
        B.X = mov_oracle(e, B, np.zeros(n))
        br_mean = e.calculate_mean()
        BR.append(br_mean)

        file_BR.write(str(br_mean) + ', ')
        for file in [file_X, file_BR, file_Y, file_XA, file_XB]:
            file.flush()
        print("means: {}, {}".format(ftpl_mean, br_mean))
    file.write("X: {}".format(X))
    file.write("Y: {}".format(Y))
    file.write("BR: {}".format(BR))

    file.write("X_A: {}".format(XAs))
    file.write("X_B: {}".format(XBs))
    file.close()
    return X, np.array(Y)-np.array(BR)

def main():
    Xs, Ys = [], []
    X, Y = run(6)
    Xs.append(X)
    Ys.append(Y)
    X_mean = np.mean(Xs, axis=0)
    Y_mean = np.mean(Ys, axis=0)
    print(X_mean, Y_mean)
    # plt.scatter(X, Y)
    # plt.show()
    # plt.savefig('ftpl.png', dpi=300)

main()
