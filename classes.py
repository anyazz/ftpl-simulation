import random
import numpy as np
import math
from utils import *
np.set_printoptions(threshold=np.inf)

delta = 0.0001

class Election:
    def __init__(self, data, n, candidates, T, opinion_attr=None, theta=[], rand=False, model_type='linear'):
        self.data = data  
        self.P = np.array(data["trustMatrix"])
        if n < len(self.P):
            idx = np.array(random.sample(range(len(self.P)), n))
            self.P = self.P[idx[:, None], idx]
            for i, row in enumerate(self.P):
                s = sum(row)
                self.P[i] *= 1./s
            print(self.P)
            for row in self.P:
                assert(abs(sum(row) - 1) < 0.01)
        self.n = n
        self.A, self.B = candidates
        self.A.opp, self.B.opp = self.B, self.A
        self.P_T = np.linalg.matrix_power(self.P, T)
        self.alpha = np.sum(np.array(self.P_T), axis=0)
        self.theta = np.array([0.5] * self.n)
        self.assign_opinions(theta, model_type, opinion_attr, rand)

    def assign_opinions(self, theta, model_type, opinion_attr, rand):
        if len(theta):
            self.theta = np.array(theta)
        else:
            if model_type == 'linear':
                if rand:
                    self.theta = np.array([random.random() for _ in range(self.n)])
                else:
                    self.attribute_opinions(opinion_attr)                

    def attribute_opinions(self, attr):
        # r, s, d = (0, 0.5), (0.4, 0.6) , (0.5, 1)
        ranges = [(0, 0.3), (0.2, 0.5) , (0.35, 0.6), (0.4, 0.65), (0.5, 0.8) , (0.7, 1)]
        # random.shuffle(ranges)
        rule = {
        # "race": {0: s, 1: r, 2: d, 3: s, 4: s, 5: s},
        "sex": {1: ranges[0], 2: ranges[-1], 0: (0.5, 0.5)},
        # "grade": {7: ranges[0], 8: ranges[1], 9:ranges[5], 10:ranges[3], 11:ranges[4], 12:ranges[2], 0: (0.5, 0.5)}
        }

        for i in range(self.n):
            a = self.data[attr][i][0]
            theta_range = rule[attr].get(a, (0.5, 0.5))
            # self.theta[i] = theta_range
            self.theta[i] = random.uniform(theta_range[0], theta_range[1])
        self.theta = np.array(self.theta)

    def advertise(self):
        theta_0 = self.theta + (np.ones(self.n) - self.theta) * self.A.p * self.A.X - self.theta * self.B.p * self.B.X \
        + (2 * self.theta - np.ones(self.n)) * self.A.p * self.B.p * self.A.X * self.B.X
        return theta_0

    def update_network(self):
        self.theta_0 = self.advertise()
        self.theta_T = np.matmul(self.P_T, self.theta_0)

    def calculate_mean(self):
        self.update_network()
        mean = np.sum(self.theta_T)
        return mean


class Candidate:
    def __init__(self, id_, k, goal, n, p=[]):
        self.id = id_
        self.goal = goal
        self.k = k
        if len(p):
            self.p = np.array(p) 
        else:
            self.p = [random.random() for _ in range(n)]
        self.X = np.zeros(n)
        self.ftpl_history = []
        self.opp = None

    # def assign_p(self):


    # u
    def marginal_payoff(self, e, X_opp):
        # set sign to negative for B
        sign = 1 if self.goal else -1
        self.u = sign * e.alpha * ((np.array([self.goal] * e.n) - e.theta) * self.p + (2 * e.theta - np.ones(e.n)) * self.p * self.opp.p * X_opp)
        return self.u
    
    # eq. 3.11: expenditure required to convert node to 1
    def max_expenditure(self, e, X_opp, i):
        num = self.goal - e.theta[i] + (e.theta[i] - (self.opp.goal)) * self.opp.p[i] * X_opp[i]
        denom = (self.goal-e.theta[i]) * self.p[i] + (2 * e.theta[i] - 1) * self.p[i] * self.opp.p[i] * X_opp[i]
        if denom == 0:
            return 0
        return min(num/denom, 1/self.p[i])




