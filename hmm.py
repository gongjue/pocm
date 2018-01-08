import numpy as np
import util

class HMM:
    """
    Multiple subjects, multiple actions
    A: different across actions. nActions x nStates x nStates
    B: shared by different actions. nStates x nLevels
    """
    def __init__(self, pi, A, B):
        """
        Initialize HMM, with initial value of transition, emission and belief
        :param A: initial transition
        :param B: initial emission
        :param pi: initial belief
        """
        self.A = A
        self.B = B
        self.pi = pi
        self.nStates, self.nLevels = self.B.shape
        self.nActions = self.A.shape[0]

    def evolve(self, init_state, Actions):
        """
        Let the HMM run for some time
        :param init_state: initial state
        :param T: length of running
        :return: observations: a list of observations
        """
        assert 0 <= init_state < self.nStates, "invalid initial state"
        nSubjects = Actions.shape[0]
        nSamples = Actions.shape[1]
        States = np.zeros((nSubjects,nSamples))
        Observations = np.zeros((nSubjects,nSamples))
        for i in range(nSubjects):
            States[i, 0] = init_state
            Observations[i, 0] = np.random.choice(self.nLevels, p=self.B[init_state])
            _state = init_state
            for t in range(1, nSamples):
                _state = np.random.choice(self.nStates, p=self.A[Actions[i, t-1]][_state])
                States[i, t] = _state
                Observations[i, t] = np.random.choice(self.nLevels, p=self.B[_state])
        return Observations.astype(int), States.astype(int)

    def viterbi(self, observations, pid=None):
        nSamples = len(observations)
        delta = np.zeros((nSamples, self.nStates))
        pre = np.zeros((nSamples, self.nStates))
        path = [-1] * nSamples
        for s in range(self.nStates):
            _pi_ = self.pi[pid] if pid is not None else self.pi
            delta[0, s] = _pi_[s] * self.B[s, observations[0]]
        for t in range(1, nSamples):
            for s in range(self.nStates):
                delta[t, s] = np.max(delta[t - 1] * self.A[:, s]) * self.B[s, observations[t]]
                pre[t, s] = np.argmax(delta[t - 1] * self.A[:, s])
        path[-1] = np.argmax(delta[-1])
        for t in range(nSamples - 2, -1, -1):
            path[t] = int(pre[t + 1, path[t + 1]])
        return path

    def viterbi_count(self, Observations):
        nSubjects, nSamples = Observations.shape
        Path = np.array([self.viterbi(Observations[i], pid=i) for i in range(nSubjects)])
        newpi = np.zeros(self.nStates)
        newA = np.zeros((self.nStates, self.nStates))
        newB = np.zeros((self.nStates, self.nLevels))
        for i in range(nSubjects):
            newpi[Path[i, 0]] += 1
            for t in range(nSamples):
                if t > 0:
                    newA[Path[i, t - 1], Path[i, t]] += 1
                newB[Path[i, t], Observations[i, t]] += 1
        newpi = util.normalize(newpi + 1e-4, axis=0)
        newA = util.normalize(newA + 1e-4, axis=1)
        newB = util.normalize(newB + 1e-4, axis=1)
        return newpi, newA, newB

    def update(self, Observations, Actions, Xi, Gamma):
        nSubjects = Observations.shape[0]
        newpi = Gamma[:, :, 0]
        newAnumer = np.zeros((self.nActions, self.nStates, self.nStates))
        newAdenom = np.zeros((self.nActions, self.nStates, self.nStates))
        for i in range(nSubjects):
            for act in range(self.nActions):
                mask_act = Actions[i, :-1] == act
                gamma_ = Gamma[i][:, :-1]
                newAnumer[act] += np.sum(Xi[i][:, :, mask_act], axis=2)
                newAdenom[act] += np.sum(gamma_[:, mask_act], axis=1, keepdims=True)
        newA = util.elim_zero(newAnumer) / util.elim_zero(newAdenom)

        newBnumer = np.zeros((self.nStates, self.nLevels))
        newBdenom = np.zeros((self.nStates, self.nLevels))
        for i in range(nSubjects):
            for lev in range(self.nLevels):
                mask_lev = Observations[i] == lev
                newBnumer[:, lev] += np.sum(Gamma[i][:, mask_lev], axis=1)
                newBdenom[:, lev] += np.sum(Gamma[i], axis=1)
        newB = util.elim_zero(newBnumer) / util.elim_zero(newBdenom)
        return newpi, newA, newB

    def learn(self, Observations, Actions, pi0=None, A0=None, B0=None, trueA=None, trueB=None, nIter=1000, criterion=1e-3, print_freq=-1):
        nSubjects, nSamples = Observations.shape
        Xi = np.zeros((nSubjects, self.nStates, self.nStates, nSamples - 1))
        Gamma = np.zeros((nSubjects, self.nStates, nSamples))
        RECORD = {'loglikelihood': [], 'A': [], 'B': [], 'dist_A': None, 'dist_B': None}

        itr = 0
        done = 0
        LL, DISTA, DISTB = [], [], []
        if pi0 is not None:
            pi, A, B = pi0, A0, B0
        else:
            pi, A, B = self.pi, self.A, self.B
        while not done:
            ll = []
            for i in range(nSubjects):
                alpha, beta = util.forward_backward(Observations[i], Actions[i], pi[i], A, B)
                ll.append(np.log(alpha[:, -1].sum()))
                xi, gamma = util.exp_counts(Observations[i], Actions[i], A, B, alpha, beta)
                Xi[i] = xi
                Gamma[i] = gamma
            newpi, newA, newB = self.update(Observations, Actions, Xi, Gamma)
            err1 = np.max(np.abs(pi - newpi))
            err2 = np.max(np.abs(A - newA))
            err3 = np.max(np.abs(B - newB))
            LL.append(ll)
            if trueA is not None:
                distA = 0
                for a in range(self.nActions):
                    distA += util.matrix_distance(newA[a], trueA[a])
                distB = util.matrix_distance(newB, trueB)
                DISTA.append(distA)
                DISTB.append(distB)
            # update RECORD
            RECORD['A'].append(newA)
            RECORD['B'].append(newB)
            if print_freq > 0:
                if (itr + 1) % print_freq == 0:
                    print("%4d %.4f %.4f %.4f %.4f " % (itr + 1, err1, err2, err3, np.mean(ll)), end=" ")
                    if trueA is not None:
                        print("%.4f" % (distA), end=" ")
                        print("%.4f" % (distB), end=" ")
                    print()
            A[:], B[:], pi[:] = newA, newB, newpi

            if itr > nIter:
                done = 1
            if err1 < criterion and err2 < criterion and err3 < criterion:
                done = 1
            itr += 1
        RECORD['loglikelihood'] = np.array(LL).T
        RECORD['dist_A'] = np.array(DISTA).T
        RECORD['dist_B'] = np.array(DISTB).T
        return pi, A, B, RECORD





def hmm_test_1():
    A = np.array([
    [[0.5,0.5],
     [0.5,0.5]]])
    B = np.array([
    [0.4,0.1,0.5],
    [0.1,0.5,0.4]])
    pi = np.array([[0.5,0.5]])
    Actions = np.zeros((1,20)).astype(int)
    hmm = HMM(pi, A, B)
    Observations, _ = hmm.evolve(0, Actions)
    pi, A, B, RECORD = hmm.learn(Observations, Actions, None, None, 1000, 1e-6)
    print("pi\n", np.array_str(pi, precision=4, suppress_small=True))
    print("A \n", np.array_str(A, precision=4, suppress_small=True))
    print("B \n", np.array_str(B, precision=4, suppress_small=True))



def hmm_test_2(nSubjects, nStates, nLevels, nSamples, nActions):
    A = util.normalize(np.random.random(size=(nActions, nStates, nStates)), axis=2)
    B = util.normalize(np.random.random(size=(nStates, nLevels)), axis=1)
    pi = util.normalize(np.random.random(size=(nSubjects, nStates)), axis=1)
    hmm = HMM(pi, A, B)
    action_dist = util.normalize(np.random.random(size=(nActions,)), axis=0)
    # observation_dist = normalize(np.random.random(size=(nLevels,)),axis=0)
    Actions = np.random.choice(nActions, [nSubjects, nSamples], p=action_dist).astype(int)
    Observations, _ = hmm.evolve(0, Actions)
    # A0 = util.normalize(np.random.random(size=(nActions, nStates, nStates)), axis=2)
    # B0 = util.normalize(np.random.random(size=(nStates, nLevels)), axis=1)
    pi, A, B, RECORD = hmm.learn(Observations, Actions, pi, A, B, A, B, 1000, 1e-4, 10)
    print("pi\n", np.array_str(pi, precision=4, suppress_small=True))
    print("A \n", np.array_str(A, precision=4, suppress_small=True))
    print("B \n", np.array_str(B, precision=4, suppress_small=True))
    print(RECORD['loglikelihood'])


if __name__ == "__main__":
    np.random.seed(124)
    # hmm_test_1()
    hmm_test_2(nSubjects=10, nStates=3, nLevels=3, nSamples=100, nActions=1)