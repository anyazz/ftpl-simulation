import random
import numpy as np
import math
from utils import roundl
    
def mov_oracle(e, cand, X_opp):
    X = np.zeros(e.n)
    remaining = cand.k
    score = cand.marginal_payoff(e, X_opp)
    heap = [(-(score[i]), i) for i in range(len(score))]
    heapq.heapify(heap)
    while remaining > 0 and len(heap):
        if len(heap):
            x_score, x = heapq.heappop(heap)
            max_X = cand.max_expenditure(e, X_opp, x)
            X[x] = min(max_X, remaining)
            assert X[x] >= 0
        remaining -= X[x]
    return X
