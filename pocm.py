import numpy as np
import util


class POCM:

    def __init__(self, pi, A, B, C):
        self.pi = pi
        self.A = A
        self.B = B
        self.C = C
        self.nGroups, self.nStates, self.nLevels = self.B.shape
        self.nActions = self.A.shape[1]

    def update1(self, Observations, Actions, Xi, Gamma):
        nSubjects, nSamples = Observations.shape
        newpi = np.zeros((self.nGroups, self.nStates))

        pi, A, B, C = self.pi, self.A, self.B, self.C
        for k in range(self.nGroups):
            for s in range(self.nStates):
                for i in range(nSubjects):
                    numer = C[i, k] * Gamma[i, s, 0] * pi[k, s]
                    denom = C[i].dot(pi[:, s])
                    newpi[k, s] += numer / denom if denom > 1e-6 else 1e-6
        newpi = util.normalize(newpi, axis=1)

        newA = np.zeros((self.nGroups, self.nActions, self.nStates, self.nStates))
        for k in range(self.nGroups):
            for s1 in range(self.nStates):
                for s2 in range(self.nStates):
                    for i in range(nSubjects):
                        for t in range(nSamples - 1):
                            for a in range(self.nActions):
                                if Actions[i, t] == a:
                                    numer = C[i, k] * Xi[i, s1, s2, t] * A[k, a, s1, s2]
                                    denom = C[i].dot(A[:, a, s1, s2])
                                    newA[k, a, s1, s2] += numer / denom if denom > 1e-10 else 1e-10
        newA = util.normalize(newA, axis=3)

        newB = np.zeros((self.nGroups, self.nStates, self.nLevels))
        for k in range(self.nGroups):
            for s in range(self.nStates):
                for o in range(self.nLevels):
                    for i in range(nSubjects):
                        numer = C[i, k] * (Gamma[i, s] * (Observations[i] == o)).sum() * B[k, s, o]
                        denom = C[i].dot(B[:, s, o])
                        newB[k, s, o] += numer / denom if denom > 1e-10 else 1e-10
        newB = util.normalize(newB, axis=2)
        # newA, newB = A, B
        return newpi, newA, newB

    def update2(self, Observations, Actions, W, mu, Xi, Gamma):
        D = np.diag(W.sum(axis=1))
        L = D - W
        d = mu * L.dot(self.C)

        nSubjects, nSamples = Observations.shape
        pi, A, B, C = self.pi, self.A, self.B, self.C

        newC = np.zeros((nSubjects, self.nGroups))
        for i in range(nSubjects):
            for k in range(self.nGroups):
                # g = np.sum(Gamma[i][:,0]) + np.sum(Xi[i])
                numer = 0
                for s in range(self.nStates):
                    numer += pi[k, s] * Gamma[i, s, 0]  # / pi[s].dot(C[i])
                for t in range(nSamples - 1):
                    for s1 in range(self.nStates):
                        for s2 in range(self.nStates):
                            numer += A[k, Actions[i, t], s1, s2] * Xi[
                                i, s1, s2, t]  # / A[:,Actions[i,t],s1,s2].dot(C[i])
                for t in range(nSamples):
                    for s in range(self.nStates):
                        numer += Gamma[i, s, t] * B[k, s, Observations[i, t]]  # / B[:,s,Observations[i,t]].dot(C[i])
                # print(c, d[i,k])
                numer += 2 * mu * D.dot(C)[i].dot(C[i]) + 2 * mu * W.dot(C)[i, k]
                denom = self.nStates * (nSamples - 1) + 1 + nSamples + 2 * mu * W.dot(C)[i].dot(C[i]) + 2 * mu * D.dot(C)[
                   i, k]
                # print("%.6f" %cc, end=" ")
                newC[i, k] = C[i, k] * numer / denom
                # print("")
                # newC[i] = newC[i] - np.min(newC[i])+0.01
        # print(newC)
        newC = util.normalize(newC, axis=1)
        return newC

    def learn(self, Observations, Actions, W, mu, trueA=None, trueB=None,
              onlyC=0, criterion=1e-3, nIter=None, print_freq=-1):
        nSubjects, nSamples = Observations.shape
        Xi = np.zeros((nSubjects, self.nStates, self.nStates, nSamples - 1))
        Gamma = np.zeros((nSubjects, self.nStates, nSamples))
        RECORD = {'pi': [], 'A': [], 'B': [], 'C': [],
                  'loglikelihood': None, 'Q': None,
                  'dist_A': None, 'dist_B': None,
                  'bias_A': None, 'bias_B': None,
                  'KL_A': None, 'KL_B': None}

        D = np.diag(W.sum(axis=1))
        L = D - W

        itr = 0
        done = 0
        err1, err2, err3, err4 = 0, 0, 0, 0
        LL, Q, DISTA, DISTB, BIASA, BIASB, KLA, KLB = [], [], [], [], [], [], [], []
        while not done:
            ll, q = [], []
            for i in range(nSubjects):
                pi_ = util.distribute(self.C[i], self.pi)
                A_ = util.distribute(self.C[i], self.A)
                B_ = util.distribute(self.C[i], self.B)
                alpha, beta = util.forward_backward(Observations[i], Actions[i], pi_, A_, B_)
                ll.append(np.log(util.elim_zero(alpha[:, -1].sum())))
                xi, gamma = util.exp_counts(Observations[i], Actions[i], A_, B_, alpha, beta)
                Xi[i] = xi
                Gamma[i] = gamma
            if not onlyC:
                newpi, newA, newB = self.update1(Observations, Actions, Xi, Gamma)
                err1 = np.max(np.abs(self.pi - newpi))
                err2 = np.max(np.abs(self.A - newA))
                err3 = np.max(np.abs(self.B - newB))
                self.A[:], self.B[:], self.pi[:] = newA, newB, newpi

                RECORD['pi'].append(newpi)
                RECORD['A'].append(newA)
                RECORD['B'].append(newB)

            for i in range(nSubjects):
                pi_ = util.distribute(self.C[i], self.pi)
                A_ = util.distribute(self.C[i], self.A)
                B_ = util.distribute(self.C[i], self.B)
                alpha, beta = util.forward_backward(Observations[i], Actions[i], pi_, A_, B_)
                # ll.append(np.log(alpha[:, -1].sum()))
                xi, gamma = util.exp_counts(Observations[i], Actions[i], A_, B_, alpha, beta)
                Xi[i] = xi
                Gamma[i] = gamma
            newC = self.update2(Observations, Actions, W, mu, Xi, Gamma)

            err4 = np.max(np.abs(self.C - newC))
            self.C[:] = newC
            regularity = mu * np.trace(self.C.T.dot(L).dot(self.C))
            for i in range(nSubjects):
                _q = np.dot(Gamma[i, :, 0], np.log(util.elim_zero(util.distribute(self.C[i], self.pi))))  # for pi
                _q += np.sum(Xi[i].sum(axis=2) * np.log(util.elim_zero(util.distribute(self.C[i], self.A))))  # for A
                for omega in range(self.nLevels):
                    _q += np.sum(np.sum(Gamma[i, :, Observations[i] == omega]) *
                                 np.log(util.elim_zero(util.distribute(self.C[i], self.B[:, :, omega]))))  # for B
                q.append(_q - regularity / nSubjects)  # regularization term
            LL.append(ll)
            Q.append(q)
            if trueA is not None:
                distA, distB, biasA, biasB, klA, klB = [], [], [], [], [], []
                for i in range(nSubjects):
                    distA.append(util.matrix_distance(util.distribute(self.C[i], self.A[:,0,:,:]), trueA(i)))
                    distB.append(util.matrix_distance(util.distribute(self.C[i], self.B), trueB(i)))
                    biasA.append(util.matrix_bias(util.distribute(self.C[i], self.A[:,0,:,:]), trueA(i)))
                    biasB.append(util.matrix_bias(util.distribute(self.C[i], self.B), trueB(i)))
                    klA.append(util.matrix_KL(util.distribute(self.C[i], self.A[:,0,:,:]), trueA(i)))
                    klB.append(util.matrix_KL(util.distribute(self.C[i], self.B), trueB(i)))
                DISTA.append(distA)
                DISTB.append(distB)
                BIASA.append(biasA)
                BIASB.append(biasB)
                KLA.append(klA)
                KLB.append(klB)

            RECORD['C'].append(newC)
            if print_freq > 0:
                if (itr + 1) % print_freq == 0:
                    print("%3d %.4f %.4f %.4f %.4f; %.4f %.4f %.4f %.4f"
                          % (itr + 1, err1, err2, err3, err4,
                             np.mean(ll), np.mean(ll) - regularity / nSubjects,
                             np.mean(q) + regularity / nSubjects, np.mean(q)), end=" ")
                    if trueA is not None:
                        print("%.4f %.4f %.4f %.4f %.4f %.4f" %(np.mean(distA), np.mean(biasA), np.mean(klA),
                                                                np.mean(distB), np.mean(biasB), np.mean(klB)))
                    else:
                        print()
            if itr > nIter:
                done = 1
            if max(err1, err2, err3, err4) < criterion:
                done = 1
            itr += 1
        RECORD['loglikelihood'] = np.array(LL).T
        RECORD['Q'] = np.array(Q).T
        RECORD['dist_A'] = np.array(DISTA).T
        RECORD['dist_B'] = np.array(DISTB).T
        RECORD['bias_A'] = np.array(BIASA).T
        RECORD['bias_B'] = np.array(BIASB).T
        RECORD['KL_A'] = np.array(KLA).T
        RECORD['KL_B'] = np.array(KLB).T
        return RECORD

def pocm_test_1():
    A = util.normalize(np.random.rand(2,2,2,2), axis=3)
    B = util.normalize(np.random.rand(2,2,3), axis=2)
    pi = np.array([[0.2,0.8],[0.4,0.6]])
    C = np.array([[0.2,0.8],[0.9,0.1],[0.3,0.7]])
    pocm = POCM(pi,A,B,C)
    Actions = np.array(
        [[1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,0,1],
         [0,0,1,1,1,1,0,0,0,0,1,1,0,0,0,1,1,0,0,1],
         [0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,0,0,0]]).astype(int)
    Observations = np.array(
        [[2,0,0,2,1,2,1,1,1,2,1,1,1,1,1,2,2,0,0,1],
         [2,1,2,1,1,1,2,1,0,0,1,1,2,1,2,1,1,2,0,1],
         [2,1,1,1,2,1,0,1,1,2,2,0,1,1,1,2,2,0,0,0]]).astype(int)
    W = np.array([
        [1.0,0.4,0.3],
        [0.4,1.0,0.8],
        [0.3,0.8,1.0]])
    mu = 0.5
    _ = pocm.learn(Observations, Actions, W, mu, trueA=None, trueB=None, onlyC=0, criterion=1e-6, nIter=1000)
    print("pi\n", np.array_str(pocm.pi,precision=4, suppress_small=True))
    print("A \n", np.array_str(pocm.A, precision=4, suppress_small=True))
    print("B \n", np.array_str(pocm.B, precision=4, suppress_small=True))
    print("C \n", np.array_str(pocm.C, precision=4, suppress_small=True))


def pocm_test_2(nSubjects,nGroups,nStates,nLevels,nSamples,nActions,mu):
    A = util.normalize(np.random.random(size=(nGroups, nActions, nStates, nStates)), axis=3)
    B = util.normalize(np.random.random(size=(nGroups, nStates, nLevels)), axis=2)
    pi = util.normalize(np.random.random(size=(nGroups, nStates)), axis=1)
    C = util.normalize(np.random.random(size=(nSubjects, nGroups)), axis=1)
    pocm = POCM(pi, A, B, C)
    action_dist = util.normalize(np.random.random(size=(nActions,)), axis=0)
    observation_dist = util.normalize(np.random.random(size=(nLevels,)), axis=0)
    Actions = np.random.choice(nActions, [nSubjects, nSamples], p=action_dist).astype(int)
    Observations = np.random.choice(nLevels, [nSubjects, nSamples], p=observation_dist).astype(int)
    # w = np.random.rand(nSubjects, nSubjects)
    # W = (w + w.T)/2
    W, _ = util.similarity_matrix2(C, 100, 50)
    print(Observations)
    print(Actions)
    print(pi)
    print(A)
    print(B)
    print('mu =', mu)
    RECORD = pocm.learn(Observations, Actions, W, mu, trueA=None, trueB=None,
                        onlyC=0, criterion=1e-4, nIter=1000, print_freq=10)
    print("pi\n", np.array_str(pocm.pi, precision=4, suppress_small=True))
    print("A \n", np.array_str(pocm.A, precision=4, suppress_small=True))
    print("B \n", np.array_str(pocm.B, precision=4, suppress_small=True))
    print("C \n", np.array_str(pocm.C, precision=4, suppress_small=True))

    plt.figure()
    plt.plot(RECORD['loglikelihood'].mean(axis=0))
    plt.figure()
    plt.plot(RECORD['Q'].mean(axis=0))
    plt.show()
    plt.show()


if __name__ == "__main__":
    np.random.seed(12)
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('paper')
    sns.set_style('white')
    np.set_printoptions(precision=4, suppress=True)
    # pocm_test_1()
    pocm_test_2(nSubjects=20, nGroups=2, nStates=3, nLevels=3, nSamples=10, nActions=2, mu=0.01)