"""
Code by Parthe Pandit
DSC 180 Capstone
Advisor: Mikhail Belkin
"""
import numpy as np, math
import nltk

n, noise_std, gamma, p, lam = 50, 0.1, 10, 8, 1e-8


def target_poly(x):
    return x**5 - 3 * x**4


target_xsinx = lambda x: x * np.sin(x)
# def target_xsinx(x):
#     return x * np.sin(x)


def K_gauss(x, z, gamma=gamma):
    return np.exp(-np.linalg.norm(x - z) ** 2 * gamma / 2)


def K_gauss_mat(x, z, gamma=gamma):
    return np.exp(
        -np.linalg.norm(x[:, None, :] - z[None, :, :], axis=-1) ** 2 * gamma / 2
    )


# K_gauss = lambda x, z, gamma=gamma: np.exp(-np.linalg.norm(x-z)**2 * gamma/2)

K_poly = lambda x, z, gamma=gamma, p=p: (1 + np.sum(x * z) / gamma) ** p


def K_poly_mat(x, z, gamma=gamma, p=p):
    """
    if x is (m, d) and z is (n, d)
    then output is (m, n)
    """
    return (1 + np.sum(x[:, None, :] * z[None, :, :], axis=-1) / gamma) ** p


Phi_poly = lambda X, p: np.hstack([X**i for i in range(p + 1)])

Phi_poly_scaled = lambda X, p, gamma=gamma: (
    Phi_poly(X, p)
    * np.array([np.sqrt(math.comb(p, i) / gamma**i) for i in range(p + 1)])
)

K_laplace = lambda x, z, gamma=gamma: np.exp(-np.linalg.norm(x - z, axis=-1) * gamma)


def K_laplace_mat(x, z, gamma=gamma):
    return np.exp(
        -np.linalg.norm(x[:, None, :] - z[None, :, :], ord=1, axis=-1) * gamma
    )


# RFM utils


def mnorm(x, z, M, power=1, squared=True):  # (n, d), (m,d), (d,d) --> (n, m)
    M_alt = M**power
    # implements |x-z|_M^2 between pairs from x and z
    x_norm = ((x @ M_alt) * x).sum(axis=1, keepdims=True)
    if x is z:
        z_norm = x_norm
    else:
        z_norm = ((z @ M_alt) * z).sum(axis=1, keepdims=True)

    z_norm = z_norm.reshape(1, -1)

    distances = (x @ (M_alt @ z.T) * -2) + x_norm + z_norm
    if not squared:
        distances = np.sqrt(np.clip(distances, 0, np.inf))
    return distances


def K_M(x, z, M, L, power=1):
    pairwise_distances = mnorm(x, z, M, power, squared=False)
    pairwise_distances = np.clip(pairwise_distances, 0, np.inf)
    return np.exp(pairwise_distances * -(1.0 / L))


def grad_laplace_mat(X, sol, L, P, power=1, batch_size=2, norm_control=False):
    M = 0.0

    num_samples = 20000
    indices = np.random.randint(len(X), size=num_samples)

    if len(X) > len(indices):
        x = X[indices, :]
    else:
        x = X

    K = K_M(X, x, P, L, power)

    dist = mnorm(X, x, P, power, squared=False)
    dist = np.where(dist < 1e-4, np.zeros(1, dtype=np.float64), dist)

    K = K / dist
    K[K == float("inf")] = 0.0

    # sol = sol[:, None]
    a1 = sol

    n, d = X.shape
    n, c = sol.shape
    m, d = x.shape

    a1 = a1.reshape(n, c, 1)
    X1 = (X @ P).reshape(n, 1, d)
    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c * d)

    step2 = K.T @ step1
    del step1

    step2 = step2.reshape(-1, c, d)

    a2 = sol.T
    step3 = (a2 @ K).T

    del K, a2

    step3 = step3.reshape(m, c, 1)
    x1 = (x @ P).reshape(m, 1, d)
    step3 = step3 @ x1

    G = (step2 - step3) * -1 / L

    M = 0.0

    bs = batch_size
    batches = np.split(G, bs)
    for i in range(len(batches)):
        grad = batches[i]
        if norm_control:
            grad = rsvd_norm_control(grad)
        gradT = np.swapaxes(grad, 1, 2)
        M += np.sum(gradT @ grad, axis=0)
        del grad, gradT
    M /= len(G)

    return M


def rsvd_norm_control(J, keep_p=0.1):
    """
    Apply randomized svd to J and return the reconstructed matrix using the top keep_p values.
    """
    u, s, v = np.linalg.svd(J, full_matrices=False)
    s = np.diag(s)
    s = s[: int(keep_p * len(s)), : int(keep_p * len(s))]
    u = u[:, : int(keep_p * len(u))]
    v = v[: int(keep_p * len(v)), :]
    return u @ s @ v

"""
def grad_laplace_mat(a, x, z, M, L): # (n, d), (m, d), (d, d) --> (n, m, d)
    num_samples = 20000
    
    dist = mnorm(x, z, M, squared=False)
    dist = np.where(dist < 1e-4, np.zeros(1, dtype=np.float16), dist)
    
    K = K_M(x, z, M, L)/dist
    K[K == float('inf')] = 0.
    
    a = a[:, None]
    
    n, d = x.shape
    n, c = a.shape
    m, d = z.shape
    
    a = a.reshape(n, c, 1)
    X1 = (x @ M).reshape(n, 1, d)
    step1 = (a @ X1).reshape(-1, c*d)
    
    step2 = (K.T @ step1).reshape(-1, c, d)
    del step1
    
    step3 = ((a.T @ K).T).reshape(m, c, 1)
    z1 = (z @ M).reshape(m, 1, d)
    step3 = step3 @ z1
    
    G = (step2 - step3) * -1/L
    
    return G
"""


def mse(y_true, y_pred, squared=True):
    if squared:
        power = 2
    else:
        power = 1
    return ((y_true - y_pred) ** power).mean()


# text generation utils
def bleu_score(y_true, y_pred, n=4):
    return nltk.translate.bleu_score.sentence_bleu(
        y_true, y_pred, weights=(1 / n,) * n
    )

def entropy(y):
    """Computes the entropy of an array of numbers."""
    y = y / y.sum()
    logged = np.where(y > 0, np.log2(y + 1e-23), 0)
    return -np.sum(y * logged)

def perplexity(y):
    """Computes the 2 ^ entropy of an array of numbers."""
    return 2 ** entropy(y)

def softmax(X, axis=0):
    """
    Applies softmax acros specified axis.
    """
    e_x = np.exp(X - np.max(X, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)