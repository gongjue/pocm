import numpy as np
# import scipy.stats
from sklearn import cluster
from scipy.spatial import distance


def elim_zero(a, epsilon=1e-6):
    if type(a) is np.ndarray:
        a[a < epsilon] = epsilon
    elif a < epsilon:
        a = epsilon
    return a


def distribute(c, M):
    return np.array([M[i] * c[i] for i in range(len(c))]).sum(axis=0)


def normalize(A, axis):
    return A / A.sum(axis=axis, keepdims=True)


def safe_normalize(A, axis):
    """
    Normalize the matrix A along axis
    if sums up to 0, keep the same
    """
    A_ = A.sum(axis=axis, keepdims=True)
    A_[A_ == 0] = 1
    return A / A_


def matrix_distance(A, B):
    return np.sqrt(np.trace(np.dot(A-B, (A-B).T)))  # Frobenius Norm


def matrix_bias(A, B):
    return np.sum(np.abs(A-B))


def matrix_KL(A, B):
    pk = elim_zero(A.reshape(-1))
    qk = elim_zero(B.reshape(-1))
    return np.sum(pk * np.log(pk / qk), axis=0)


def forward_backward(observations, actions, pi, A, B):
    nStates = B.shape[0]
    nSamples = len(observations)
    alpha = np.zeros((nStates, nSamples))
    alpha[:, 0] = pi[:].T * B[:, observations[0]]
    for t in range(1, nSamples):
        alpha[:, t] = np.dot(alpha[:, t-1].T, A[actions[t-1]]).T * B[:, observations[t]]
    beta = np.zeros((nStates, nSamples))
    beta[:,nSamples-1] = 1.0
    for t in range(nSamples-1, 0, -1):
        beta[:, t-1] = np.dot(A[actions[t-1]], (B[:, observations[t]] * beta[:, t]))
    return alpha, beta


def exp_counts(observations, actions, A, B, alpha, beta):
    nStates = B.shape[0]
    nSamples = len(observations)
    xi = np.zeros((nStates, nStates, nSamples-1))
    for t in range(nSamples-1):
        denom = np.dot(np.dot(alpha[:, t].T, A[actions[t]]) * B[:,observations[t+1]].T, beta[:, t+1])
        for i in range(nStates):
            numer = alpha[i, t] * A[[actions[t]], i, :] * B[:, observations[t+1]].T * beta[:, t+1].T
            xi[i, :, t] = numer / denom if denom.sum() > 0 else denom
    gamma = np.squeeze(np.sum(xi,axis=1))
    # Need final gamma element for new B
    prod = (alpha[:,nSamples-1] * beta[:, nSamples-1]).reshape((-1, 1))
    if np.sum(prod) > 0:
        prod = prod / np.sum(prod)
    gamma = np.hstack((gamma,  prod))  # append one more to gamma!!!
    return xi, gamma


def likelihood_alpha(alpha):
    return np.sum(alpha[:, -1])


def likelihood_beta(beta, pi, B, o):
    return np.sum(beta[:, 0] * pi * B[:, o])


def likelihood_test(nActions, nStates, nLevels, nSamples):
    A = normalize(np.random.random(size=(nActions, nStates, nStates)), axis=1)
    B = normalize(np.random.random(size=(nStates, nLevels)), axis=1)
    pi = normalize(np.random.random(size=(nStates,)), axis=0)
    action_dist = normalize(np.random.random(size=(nActions,)), axis=0)
    observation_dist = normalize(np.random.random(size=(nLevels,)),axis=0)
    actions = np.random.choice(nActions, nSamples, p=action_dist)
    observations = np.random.choice(nLevels, nSamples, p=observation_dist)
    alpha, beta = forward_backward(observations, actions, pi, A, B)
    print('likelihood_alpha', likelihood_alpha(alpha))
    print('likelihood_beta ', likelihood_beta(beta, pi, B, observations[0]))


def similarity(x, y):
    return -np.dot((x - y), (x - y)) / 2


def similarity_matrix(X):
    n = len(X)
    W = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = similarity(X[i], X[j])
            W[i, j] = d
            W[j, i] = d
    return W


def similarity_matrix2(X, P, sigma2):
    N = X.shape[0]
    Sigma = np.cov(X)
    Z = np.random.multivariate_normal(np.zeros(N), Sigma, P).T
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = similarity(Z[i], Z[j])
            W[i, j] = d
            W[j, i] = d
    return np.exp(W / sigma2), Z


def get_membership(v2, omega):
    """
    Generate membership from three carnonical groups
    v2: predefined variance
    omega: distribution of the population among three carnonical groups
    """
    n = len(omega)
    i = np.random.choice(n, 1, p=omega)[0]
    mean = np.zeros(n)
    diag = np.ones(n)
    diag[i] = v2
    cov = np.diag(diag)
    c = np.random.multivariate_normal(mean, cov, 1)
    c = np.abs(c)[0]
    return c / c.sum()


def interaction_effect(TM, rho):
    """
    Intercation makes the stationary distribution of health larger,
    parameterized by rho
    """
    TI = TM * rho
    TI[:, 0] += 1 - rho
    return TI


def kmeans_bic(kmeans, X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels = kmeans.labels_
    # number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    # size of data set
    N, d = X.shape

    # compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2)
                                        for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    bic = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(bic)

if __name__ == '__main__':
    likelihood_test(nActions=1, nStates=2, nLevels=2, nSamples=10)