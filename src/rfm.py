import logging
import numpy as np
import matplotlib.pyplot as plt
import src.utils as utils
from src.etl import TARGET_FNS

logger = logging.getLogger(__name__)

def train_rfm(X_train, y_train, L=1., lam=1e-3, T=10):
    """
    Train an RFM kernel.
    
    Parameters
    ----------
    X_train: (n,d) numpy ndarray, training data
    y_train: (n,) numpy ndarray, true outputs
    L: float, bandwidth of kernel
    lam: float, regularization coefficient ("lambda")
    T: int, number of training iterations
    
    Returns
    -------
    alpha: (n,) numpy ndarray, contains weights for each training datapoint
    M: (d,d) numpy ndarray, contains trained weights of each feature
    """
    n, d = X_train.shape
    
    M = np.eye(d)
    for t in range(T):
        K_train = utils.K_M(X_train, X_train, M, L=L)
        alpha = y_train @ np.linalg.pinv(K_train + lam*np.eye(n))
        grad = utils.grad_laplace_mat(alpha, X_train, X_train, M, L=L)
        M = np.mean(np.swapaxes(grad, 1, 2) @ grad, axis=0)
    
    # evaluate
    y_hat = alpha @ utils.K_M(X_train, X_train, M, L)
    
    logger.info("TRAIN MSE: %.3f" % utils.mse(y_train, y_hat))
    
    return alpha, M

def test_rfm(X_train, X_test, y_test, alpha, M, L):
    """Test an RFM kernel."""
    y_hat = alpha @ utils.K_M(X_train, X_test, M, L)
    
    logger.info("TEST MSE: %.3f" % utils.mse(y_test, y_hat))

def plot_M(M, fpath=None):
    """
    Plots M.
    
    Parameters
    ----------
    M: (d,d) numpy ndarray
    
    Returns
    -------
    matplotlib plot, if fpath=None. Saves to fpath otherwise.
    """
    fig = plt.figure(figsize=(10,10))
    fig.imshow(M)
    fig.colorbar()
    
    if not fpath:
        return fig
    else:
        fig.savefig(fpath)