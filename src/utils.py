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