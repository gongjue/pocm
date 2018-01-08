import numpy as np
import util


def MC_transition_count(observations, nStates):
    N = np.zeros((nStates, nStates))
    # print(len(observations))
    for i in range(len(observations)-1):
        s1 = observations[i]
        s2 = observations[i+1]
        N[s1, s2] += 1
    return N


def MC_likelihood(pi, A, observations):
    nStates = A.shape[0]
    ll = np.log(pi[observations[0]])
    N = MC_transition_count(observations, nStates)
    ll += np.sum(N * np.log(util.elim_zero(A)))
    return ll


class CM:

    def __init__(self, pi, A, C):
        self.pi = pi  # initial belief, nGroups x nStates
        self.A = A    # transition matrix, nGroups x nStates x nStates
        self.C = C    # membership, nSubjects x nGroups
        self.nGroups, self.nStates = self.pi.shape
        self.nSubjects = self.C.shape[0]

    def update1(self, Observations):
        pi, A, C = self.pi, self.A, self.C
        # Update pi
        newpi = np.zeros((self.nGroups, self.nStates))
        for i in range(self.nSubjects):
            for k in range(self.nGroups):
                for s in range(self.nStates):
                    if Observations[i, 0] == s:
                        numer = C[i, k] * pi[k, s]
                        denom = C[i].dot(pi[:, s])
                        newpi[k, s] += numer / denom if denom > 1e-10 else 1e-10
        newpi = util.normalize(newpi, axis=1)
        # update A
        newA = np.zeros((self.nGroups, self.nStates, self.nStates))
        for i in range(self.nSubjects):
            N = MC_transition_count(Observations[i], self.nStates)
            for k in range(self.nGroups):
                for s1 in range(self.nStates):
                    for s2 in range(self.nStates):
                        numer = N[s1, s2] * C[i, k] * A[k, s1, s2]
                        denom = C[i].dot(A[:, s1, s2])
                        newA[k, s1, s2] += numer / denom if denom > 1e-10 else 1e-10
        newA = util.normalize(newA, axis=2)
        # return updated values
        return newpi, newA

    def update2(self,
                Observations, # nSubjects x nSamples
                W, # similarity matrix, nSubujects x nSubjects
                mu # tuning parameter
    ):
        nSamples = Observations.shape[1]
        D = np.diag(W.sum(axis=1))

        pi, A, C = self.pi, self.A, self.C

        newC = np.zeros((self.nSubjects, self.nGroups))
        for i in range(self.nSubjects):
            N = MC_transition_count(Observations[i], self.nStates)
            for k in range(self.nGroups):
                numer = 0
                for s in range(self.nStates):
                    if Observations[i, 0] == s:
                        numer += pi[k, s] / C[i].dot(pi[:, s])
                numer += np.sum(N * A[k] / util.elim_zero(util.distribute(C[i], A)))
                numer += mu * D.dot(C)[i].dot(C[i]) + mu * W.dot(C)[i, k]
                denom = 1 + nSamples + mu * W.dot(C)[i].dot(C[i]) + mu * D.dot(C)[i, k]
                newC[i, k] = C[i, k] * numer / denom
        newC = util.normalize(newC, axis=1)
        return newC

    def learn(self, Observations, W, mu, trueA=None, onlyC=0, criterion=1e-3, nIter=1000, print_freq=-1):
        nSubjects, nSamples = Observations.shape
        # pi, A, C = self.pi, self.A, self.C
        RECORD = {'pi': [], 'A': [], 'C': [],
                  'loglikelihood': None,
                  'dist_A': None, 'bias_A': None, 'KL_A': None}

        D = np.diag(W.sum(axis=1))
        L = D - W

        itr = 0
        done = 0
        err1, err2, err3 = 0, 0, 0
        LL, DISTA, BIASA, KLA = [], [], [], []
        while not done:
            # STEP 1
            if not onlyC:
                newpi, newA = self.update1(Observations)
                err1 = np.max(np.abs(self.pi - newpi))
                err2 = np.max(np.abs(self.A - newA))
                self.A[:], self.pi[:] = newA, newpi
                RECORD['pi'].append(newpi)
                RECORD['A'].append(newA)
            # STEP 2
            newC = self.update2(Observations, W, mu)
            err3 = np.max(np.abs(self.C - newC))
            self.C[:] = newC

            regularity = mu * np.trace(self.C.T.dot(L).dot(self.C))
            ll = []
            for i in range(nSubjects):
                ll.append(MC_likelihood(util.distribute(self.C[i], self.pi),
                                        util.distribute(self.C[i], self.A),
                                        Observations[i]))
            LL.append(ll)
            if trueA is not None:
                distA, biasA, klA = [], [], []
                for i in range(nSubjects):
                    distA.append(util.matrix_distance(util.distribute(self.C[i], self.A), trueA(i)))
                    biasA.append(util.matrix_bias(util.distribute(self.C[i], self.A), trueA(i)))
                    klA.append(util.matrix_KL(util.distribute(self.C[i], self.A), trueA(i)))
                DISTA.append(distA)
                BIASA.append(biasA)
                KLA.append(klA)

            RECORD['C'].append(newC)
            if print_freq > 0:
                if (itr + 1) % print_freq == 0:
                    print("%3d %.4f %.4f %.4f; %.4f %.4f"
                          % (itr + 1, err1, err2, err3,
                             np.mean(ll), np.mean(ll) - regularity / nSubjects), end=" ")
                    if trueA is not None:
                        print("%.4f %.4f %.4f" %(np.mean(distA), np.mean(biasA), np.mean(klA)))
                    else:
                        print()
            if itr > nIter-2:
                done = 1
            if max(err1, err2, err3) < criterion:
                done = 1
            itr += 1
        RECORD['loglikelihood'] = np.array(LL).T
        RECORD['dist_A'] = np.array(DISTA).T
        RECORD['bias_A'] = np.array(BIASA).T
        RECORD['KL_A'] = np.array(KLA).T
        return RECORD

def cm_test_1():
    A = util.normalize(np.random.rand(2, 3, 3), axis=2)
    pi = np.array([[0.2, 0.6, 0.2], [0.4, 0.4, 0.2]])
    C = np.array([[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]])
    cm = CM(pi, A, C)
    Observations = np.array(
        [[2,0,0,2,1,2,1,1,1,2,1,1,1,1,1,2,2,0,0,1],
         [2,1,2,1,1,1,2,1,0,0,1,1,2,1,2,1,1,2,0,1],
         [2,1,1,1,2,1,0,1,1,2,2,0,1,1,1,2,2,0,0,0]]).astype(int)
    W = np.array([
        [1.0,0.4,0.3],
        [0.4,1.0,0.8],
        [0.3,0.8,1.0]])
    mu = 0.01
    _ = cm.learn(Observations, W, mu, trueA=None, onlyC=0, criterion=1e-6, nIter=1000)
    print("pi\n", np.array_str(cm.pi,precision=4, suppress_small=True))
    print("A \n", np.array_str(cm.A, precision=4, suppress_small=True))
    print("C \n", np.array_str(cm.C, precision=4, suppress_small=True))


def cm_test_2(nSubjects, nGroups, nStates, nLevels, nSamples, mu, criterion):
    A = util.normalize(np.random.random(size=(nGroups, nStates, nStates)), axis=2)
    pi = util.normalize(np.random.random(size=(nGroups, nStates)), axis=1)
    C = util.normalize(np.random.random(size=(nSubjects, nGroups)), axis=1)
    cm = CM(pi, A, C)
    observation_dist = util.normalize(np.random.random(size=(nLevels,)), axis=0)
    Observations = np.random.choice(nLevels, [nSubjects, nSamples], p=observation_dist).astype(int)
    w = np.random.rand(nSubjects, nSubjects)
    W = (w + w.T)/2
    print("Observations = \n", Observations)
    print("pi = \n", pi)
    print("A = \n", A)
    print("mu = \n", mu)
    print("Learning CM ...")
    _ = cm.learn(Observations, W, mu, trueA=None, onlyC=0, criterion=criterion, nIter=1000)
    print("pi\n", np.array_str(cm.pi, precision=4, suppress_small=True))
    print("A \n", np.array_str(cm.A, precision=4, suppress_small=True))
    print("C \n", np.array_str(cm.C, precision=4, suppress_small=True))


def cm_depression():
    import simupomdp as sim
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('paper')
    sns.set_style('white')
    np.set_printoptions(precision=4, suppress=True)
    A0 = np.array([
        [[0.20, 0.50, 0.10, 0.10, 0.10],
         [0.00, 0.20, 0.50, 0.15, 0.15],
         [0.00, 0.00, 0.20, 0.50, 0.30],
         [0.00, 0.00, 0.00, 0.20, 0.80],
         [0.00, 0.00, 0.00, 0.00, 1.00]],
        [[0.56, 0.44, 0.00, 0.00, 0.00],
         [0.20, 0.65, 0.15, 0.00, 0.00],
         [0.17, 0.50, 0.33, 0.00, 0.00],
         [0.00, 0.00, 0.25, 0.25, 0.50],
         [0.00, 0.00, 0.00, 0.27, 0.73]],
        [[0.98, 0.02, 0.00, 0.00, 0.00],
         [0.05, 0.94, 0.01, 0.00, 0.00],
         [0.00, 0.19, 0.80, 0.01, 0.00],
         [0.00, 0.00, 0.38, 0.61, 0.01],
         [0.00, 0.00, 0.00, 0.43, 0.57]]
    ])
    pi0 = np.array([
        [0.80, 0.20, 0.00, 0.00, 0.00],
        [0.00, 0.00, 0.20, 0.50, 0.30],
        [0.20, 0.20, 0.20, 0.20, 0.20]
    ])
    nGroups, nStates = pi0.shape
    nSubjects = 500
    nSamples = 15
    # init_s_dist = [0.3, 0.2, 0.2, 0.15, 0.15]
    omega = np.array([0.3, 0.4, 0.3])
    noise_scale = 0.05

    def simulate_history(A, pi, T):
        ob = np.zeros(T)
        s = np.random.choice(nStates, 1, p=pi)[0]
        for i in range(T):
            s = np.random.choice(nStates, 1, p=A[s])[0]
            ob[i] = s
        return ob.astype(int)

    def generate_history(N, T, pi, A, omega):
        """Simulate observations from true matrices, N patients
        """
        data = np.zeros((N, T))
        CI = np.zeros((N, 3))
        PII = np.zeros((N, nStates))
        AI = np.zeros((N, nStates, nStates))
        for i in range(N):
            Ci = util.get_membership(5, omega)
            CI[i] = Ci
            pii = util.distribute(Ci, pi) + np.random.uniform(0.0, noise_scale, size=[nStates])
            pii = util.normalize(pii, axis=0)
            PII[i] = pii
            Ai = util.distribute(Ci, A) + np.random.uniform(0.0, noise_scale, size=[nStates, nStates])
            Ai = util.normalize(Ai, axis=1)
            AI[i] = Ai
            data[i] = simulate_history(Ai, pii, T)
        return data.astype(int), CI, PII, AI



    data, trueC, truepi, trueA = generate_history(nSubjects, nSamples, pi0, A0, omega)

    pi_init = np.zeros(nStates)
    for i in range(nSubjects):
        pi_init[data[i, 0]] += 1
    pi_init = util.normalize(pi_init, axis=0)
    pi_init = np.tile(pi_init, [nGroups, 1])
    pi_init = util.normalize(pi_init + np.random.uniform(0.0, noise_scale, size=[nGroups, nStates]), axis=1)
    A_init = util.normalize(MC_transition_count(data.reshape(-1), nStates), axis=1)
    A_init = np.tile(A_init, [nGroups, 1, 1])
    A_init = util.normalize(A_init + np.random.uniform(0.0, noise_scale, size=[nGroups, nStates, nStates]), axis=2)
    C_init = util.normalize(np.random.uniform(0.0, 1.0, size=[nSubjects, nGroups]), axis=1)
    # W = similarity_matrix2(C_init, 100, 50)
    W, _ = util.similarity_matrix2(trueC, 100, 20)
    # cm = CM(pi_init, A_init, C_init)
    cm = CM(pi0, A0, trueC)
    print("Initial guess:")
    print("pi:\n", cm.pi)
    print("A:\n", cm.A)
    print("C:\n", cm.C[:5])
    print(np.array_str(W[:10, :10], precision=2, suppress_small=True))
    mu = 0.1
    print("mu =", mu)
    print("CM training...")
    RECORD = cm.learn(data, W, mu=mu, trueA=(lambda i: trueA[i]), onlyC=0, criterion=1e-6, nIter=200, print_freq=1)
    print("Final results:")
    print("pi:\n", cm.pi)
    print("A:\n", cm.A)
    print("C:\n", cm.C[:5])

    plt.figure()
    plt.plot(RECORD['loglikelihood'].mean(axis=0))
    plt.show()



if __name__ == "__main__":
    np.random.seed(1)
    # cm_test_1()
    cm_test_2(nSubjects=100, nGroups=3, nStates=3, nLevels=3, nSamples=20, mu=0.1, criterion=1e-3)
    # cm_depression()



