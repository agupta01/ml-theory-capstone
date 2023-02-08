import logging
import numpy as np
import matplotlib.pyplot as plt
import src.utils as utils
from src.etl import TARGET_FNS

logger = logging.getLogger(__name__)


def train_rfm(X_train, y_train, power=1, L=1.0, lam=1e-3, T=10):
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
        K_train = utils.K_M(X_train, X_train, M, L, power)
        alpha = y_train @ np.linalg.pinv(K_train + lam * np.eye(n))
        M = utils.grad_laplace_mat(X_train, alpha, L, M, power)

    # evaluate
    y_hat = alpha @ utils.K_M(X_train, X_train, M, L, power)

    logger.info("TRAIN MSE: %.3f" % utils.mse(y_train, y_hat))

    return alpha, M


def test_rfm(X_train, X_test, y_test, alpha, M, L, power=1):
    """Test an RFM kernel."""
    y_hat = alpha @ utils.K_M(X_train, X_test, M, L, power)

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
    fig = plt.figure(figsize=(10, 10))
    fig.imshow(M)
    fig.colorbar()

    if not fpath:
        return fig
    else:
        fig.savefig(fpath)


def plot_results(X, y, y_hat, fpath, test=False):
    """Save prediction to path."""
    plt.scatter(X, y, label=f"{'Test' if test else 'Train'} True")
    plt.scatter(X, y_hat, label=f"{'Test' if test else 'Train'} Predicted")
    plt.legend()
    plt.savefig(fpath)
