import numpy as np
import cvxpy as cvx

import util


def set_contains_array(S, a):
    """
    :param S: list of np.ndarray
    :param a: np.ndarray
    :return: contains, 0 or 1
    """
    contains = 0
    for b in S:
        if not (a - b).any():  # if a contained in S
            contains = 1
    return contains


def set_sum_two(A, B):
    """
    :param A: list of np.ndarray
    :param B: list of np.ndarray
    :return: list of np.ndarray
    """
    C = []
    for a in A:
        for b in B:
            if not set_contains_array(C, a + b):
                C.append(a + b)
    return C


def set_sum_list(Omega):
    """
    Set sum of multiple set of np.ndarray
    :param Omega: list of list of np.ndarray
    :return: list of np.ndarray
    """
    S = Omega[0]
    # print 'len(Omega) =', len(Omega)
    # print 0, 'S =', S
    for i in range(1, len(Omega)):
        # print i, 'Omega[i] =',Omega[i]
        S = set_sum_two(S, Omega[i])
        # print i, 'S =', S
    return S


def pointwise_dominate(w, U):
    """
    Test if w is point-wise dominated by all u in U
    :param w: np.ndarray
    :param U: list of np.ndarray
    :return:
    """
    for u in U:
        if np.all(w < u):
            return True
    return False


def lp_dominate(w, U):
    """
    Computes the belief in which w improves U the most.
    With LP in White & Clark
    :param w: np.ndarray
    :param U: list of np.ndarray
    :return: b if d >= 0 else None
    """
    # print("LP dominate")
    if len(U) == 0:
        return w
    S = len(w)
    d = cvx.Variable()
    b = cvx.Variable(S)
    objective = cvx.Maximize(d)
    # print("U", U)
    constraints = [b.T*(w-u) >= d for u in U] + [np.sum(b) == 1]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    # print("d =", d.value)
    if d.value >= 0:
        return np.ravel(b.value)
    else:
        return None


def dec_dominate(w, U):
    """
    Computes the belief in which w improves U the most.
    With Bender's decomposition (Walraven & Spaan, 2017)
    :param w: np.ndarray
    :param U: list of np.ndarray
    :return: b if d >= 0 else None
    """
    if len(U) == 0:
        return w
    S = len(w)

    d = cvx.Variable()
    b = cvx.Variable(S)
    objective = cvx.Maximize(d)
    # print("U", U)
    constraints = [np.sum(b) == 1]
    b_ = np.random.random(S)
    b_ = b_ / np.sum(b_)
    U_ = []
    while 1:
        _b = b_
        u_ = U[np.argmin([np.dot((w - U[i]), _b) for i in range(len(U))])]
        constraints += [d <= b.T*(w-u_)]
        U_.append(u_)
        prob = cvx.Problem(objective, constraints)
        _ = prob.solve()
        b_ = np.ravel(b.value)
        if not (b_ - _b).any():
            break
    if d.value >= 0:
        return _b
    else:
        return None


def lex_less(u, w):
    if w is None:
        return False
    for i in range(len(u)):
        if u[i] > w[i]:
            return False
    return True


def best_point(b, U):
    # print("Find best")
    _max = -np.inf
    w = None
    for i in range(len(U)):
        u = U[i]
        # print("b", b)
        # print("u", u)
        x = np.dot(b, u)
        # print("x", x)
        if x > _max or (x == _max and lex_less(u, U[w])):
            w = i
            _max = x
            # print("max", _max)
    return w


def prune(W, A=None):
    # print("prune", W)
    D, E = [], []
    while len(W) > 0:
        w = W[-1]
        if pointwise_dominate(w, D):
            W.pop()
        else:
            # b = lp_dominate(w, D)
            b = dec_dominate(w, D)
            if b is None:
                W.pop()
            else:
                i = best_point(b, W)
                D.append(W[i])
                if A is not None:
                    E.append(A[i])
                W.pop(i)
    if A is not None:
        return D, E
    else:
        return D

def set_union(V):
    V_ = []
    for v in V:
        V_ += v
    return V_

class POMDP:

    def __init__(self, P=None, Z=None, R=None, g=None, alpha=1.0):
        self.P = P  # m x n x n: a(t)->s(t)->s(t+1)
        self.Z = Z  # m x n x k: a(t)->s(t+1)->o(t+1)
        self.R = R  # m x n x n: a(t)->s(t+1)->s(t+1)
        self.g = g  # n x 1:     s(T)
        self.alpha    = alpha  # discount factor
        self.nActions = self.Z.shape[0]  # m
        self.nStates  = self.Z.shape[1]  # n
        self.nLevels  = self.Z.shape[2]  # k
        if g is None:
            self.g = np.zeros(self.nStates)
        # print self.nActions, self.nStates, self.nLevels

    def update_belief(self, b, a, o):
        p = self.Z[a, :, o] * self.P[a].T.dot(b)
        return p / p.sum()

    def monahan_enumeration(self, V):
        """construct the set of Omega
        :param V: input list of alpha vectors
        """
        V_, A_ = [], []
        for a in range(self.nActions):
            # print("Action", a)
            Va = []
            _r = np.sum(self.P[a] * self.R[a], axis=1) / self.nLevels
            # print("_r:", _r)
            for z in range(self.nLevels):
                # print("Obs", z)
                Vaz = [_r + self.alpha * (self.Z[a,:,z] * v).dot(self.P[a]) for v in V]
                # print("Vaz", Vaz)
                if len(Va) > 0:
                    Va = prune(set_sum_two(Va, Vaz))  # incremental pruning
                else:
                    Va = Vaz
            A_ += [a for _ in Va]
            V_ += Va
        V_, A_ = prune(V_, A_)
        return V_, A_

    def transition(self, a, s):
        return np.random.choice(self.nStates, p=self.P[a, s])

    def emmission(self, a, s):
        return np.random.choice(self.nStates, p=self.Z[a, s])

    @staticmethod
    def optimal_action(b, V, A):
        assert len(V) == len(A)
        values = [np.dot(b, v) for v in V]
        opt_idx = np.argmax(values)
        return A[opt_idx], V[opt_idx]

    def solve(self, T):
        V = self.g
        Values = [None for _ in range(T)] + [[self.g]]
        Actions = [None for _ in range(T)]
        for t in range(T):
            V, A = self.monahan_enumeration(V)
            Values[T-1-t] = V
            Actions[T-1-t] = A
        return Values, Actions


    def plan(self, T, initial_belief=None, perform=False):
        V = self.g
        if initial_belief is None:
            initial_belief = np.ones(self.nStates) / self.nStates
        b = initial_belief
        Values = [None for _ in range(T)] + [[self.g]]
        Actions = [None for _ in range(T)]
        for t in range(T):
            V, A = self.monahan_enumeration(V)
            Values[T - 1 - t] = V
            Actions[T - 1 - t] = A
        a0, v0 = self.optimal_action(b, Values[0], Actions[0])
        if not perform:
            return a0, v0
        s = np.random.choice(self.nStates, p=b)
        actions, states, observations, reward = [], [], [], 0.0
        for t in range(T):
            a, v = self.optimal_action(b, Values[t], Actions[t])
            # print('a', a)
            # print('v', v)
            _s = s
            s = self.transition(a, s)
            o = self.transition(a, s)
            b = self.update_belief(b, a, o)
            states.append(_s)
            actions.append(s)
            observations.append(o)
            reward += self.R[a, _s, s] * self.alpha ** t
        return a0, v0, actions, states, observations, reward

def test_pomdp(nActions, nStates, nLevels, alpha):
    # P = np.array([
    #     [[0.25, 0.75], [0.6 , 0.4 ]],
    #     [[0.5 , 0.5 ], [0.7 , 0.3 ]]])
    # Z = np.array([
    #     [[0.55, 0.45], [0.3 , 0.7 ]],
    #     [[0.65, 0.35], [0.25, 0.75]]])
    # R = np.array([
    #     [[2., 2. ], [ 0.,  0.]],
    #     [[3., 3. ], [-1., -1.]]])
    # g = np.array([2., -1.])
    P = util.normalize(np.random.random(size=(nActions, nStates, nStates)), axis=2)
    Z = util.normalize(np.random.random(size=(nActions, nStates, nLevels)), axis=2)
    R = util.normalize(np.random.random(size=(nActions, nStates, nStates)), axis=2)
    g = util.normalize(np.random.random(size=(nStates)), axis=0)
    pomdp = POMDP(P, Z, R, g, alpha)
    T = 10
    V = pomdp.g
    a0, v0 = pomdp.plan(T, initial_belief=None, perform=False)
    # a0, v0, actions, states, observations, reward = pomdp.plan(T, initial_belief=None, perform=True)
    # print('a0 =', a0, 'v0 =', v0)
    # print('actions:', actions)
    # print('states:', states)
    # print('observations:', observations)
    # print('reward:', reward)
    # for t in range(T):
    #     print("Iteration", t+1)
    #     V, A = pomdp.monahan_enumeration(V)
    #     for v, a in zip(V, A):
    #         print(v, a)

if __name__ == "__main__":
    # import timeit
    # print(timeit.timeit("main()"))
    import time
    for s in range(123, 133):
        start_time = time.time()
        np.random.seed(s)
        print("===== SEED %d =====" %(s))
        test_pomdp(nActions=2, nStates=3, nLevels=3, alpha=0.9975)
        end_time = time.time()
        print(end_time - start_time)
