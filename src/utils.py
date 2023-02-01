"""
Code by Parthe Pandit
DSC 180 Capstone
Advisor: Mikhail Belkin
"""
import numpy as np, math

n, noise_std, gamma, p, lam = 50, 0.1, 10, 8, 1e-8

def target_poly(x):
    return x**5 - 3* x**4

target_xsinx = lambda x: x * np.sin(x)
# def target_xsinx(x):
#     return x * np.sin(x)

def K_gauss(x, z, gamma=gamma):
    return np.exp(-np.linalg.norm(x-z)**2 * gamma/2)

def K_gauss_mat(x, z, gamma=gamma):
    return np.exp(-np.linalg.norm(x[:,None,:]-z[None,:,:], axis=-1)**2 * gamma/2)

# K_gauss = lambda x, z, gamma=gamma: np.exp(-np.linalg.norm(x-z)**2 * gamma/2)

K_poly = lambda x, z, gamma=gamma, p=p: (1 + np.sum(x*z)/gamma)**p

def K_poly_mat(x, z, gamma=gamma, p=p): 
    """
    if x is (m, d) and z is (n, d)
    then output is (m, n)
    """
    return (1 + np.sum(x[:,None,:]*z[None,:,:], axis=-1)/gamma)**p

Phi_poly = lambda X, p: np.hstack([X**i for i in range(p+1)])

Phi_poly_scaled = lambda X, p, gamma=gamma: (
    Phi_poly(X, p)*
    np.array([np.sqrt(math.comb(p, i)/gamma**i) for i in range(p+1)])
)

K_laplace = lambda x, z, gamma=gamma: np.exp(-np.linalg.norm(x-z, axis=-1) * gamma)

def K_laplace_mat(x, z, gamma=gamma):
    return np.exp(-np.linalg.norm(x[:,None,:]-z[None,:,:], ord=1, axis=-1) * gamma)


# RFM utils

def mnorm(x, z, M, squared=True): # (n, d), (m,d), (d,d) --> (n, m)
    # implements |x-z|_M^2 between pairs from x and z
    x_norm = ((x @ M)*x).sum(axis=1, keepdims=True)
    if x is z:
        z_norm = x_norm
    else:
        z_norm = ((z @ M)*z).sum(axis=1, keepdims=True)
        
    z_norm = z_norm.reshape(1, -1)
    
    distances = (x @ (M @ z.T)*-2) + x_norm + z_norm
    if not squared:
        distances = np.sqrt(np.clip(distances, 0, np.inf))
    return distances

def K_M(x, z, M, L):
    pairwise_distances = mnorm(x, z, M, squared=False)
    pairwise_distances = np.clip(pairwise_distances, 0, np.inf)
    return np.exp(pairwise_distances * -(1./L))

def grad_laplace_mat(a, x, z, M, L): # (n, d), (m, d), (d, d) --> (n, m, d)
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

def mse(y_true, y_pred, squared=True):
    if squared:
        power = 2
    else:
        power = 1
    return ((y_true - y_pred)**power).mean()