import logging
import numpy as np
import matplotlib.pyplot as plt
from . import utils
from .etl import TARGET_FNS

logger = logging.getLogger(__name__)


class RFM():
    def __init__(self, L=1.0, lam=1e-3, T=10, power=1):
        self.L = L
        self.lam = lam
        self.T = T
        self.power = power

    def fit(self, X_train, y_train, L=None, lam=None, T=None, power=None, norm_control=False):
        if L is not None:
            self.L = L
        if lam is not None:
            self.lam = lam
        if T is not None:
            self.T = T
        if power is not None:
            self.power = power

        self.alpha, self.M = train_rfm(
            X_train, y_train, L=self.L, lam=self.lam, T=self.T, power=self.power, norm_control=norm_control
        )

    def predict(self, X_test):
        return utils.K_M(X_test, X_test, self.M, self.L, self.power) @ self.alpha

    def score(self, X_test, y_test):
        return utils.mse(y_test, self.predict(X_test))


def train_rfm(X_train, y_train, power=1, L=1.0, lam=1e-3, T=10, norm_control=False):
    """
    Train an RFM kernel.

    Parameters
    ----------
    X_train: (n,d) numpy ndarray, training data
    y_train: (n,m) numpy ndarray, true outputs
    L: float, bandwidth of kernel
    lam: float, regularization coefficient ("lambda")
    T: int, number of training iterations

    Returns
    -------
    alpha: (n,) numpy ndarray, contains weights for each training datapoint
    M: (d,d) numpy ndarray, contains trained weights of each feature
    """
    n, d = X_train.shape
    n, m = y_train.shape

    M = np.eye(d)
    for t in range(T):
        K_train = utils.K_M(X_train, X_train, M, L, power)
        alpha = np.linalg.solve(K_train + lam*np.eye(n), y_train)
        M = utils.grad_laplace_mat(X_train, alpha, L=L, M=M, power=power, norm_control=norm_control)

    # evaluate
    y_hat = utils.K_M(X_train, X_train, M, L, power) @ alpha # (n,n) * (n, m) -> (n,m)

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
