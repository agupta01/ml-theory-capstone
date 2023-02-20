import logging
import numpy as np
import matplotlib.pyplot as plt
from . import utils
from .etl import TARGET_FNS

logger = logging.getLogger(__name__)


class RFM:
    def __init__(self, L=1.0, lam=1e-3, T=10, power=1):
        self.L = L
        self.lam = lam
        self.T = T
        self.power = power

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        L=None,
        lam=None,
        T=None,
        power=None,
        norm_control=False,
        baseline=False,
    ):
        if L is not None:
            self.L = L
        if lam is not None:
            self.lam = lam
        if T is not None:
            self.T = T
        if power is not None:
            self.power = power

        self.X_train = X_train

        self.alpha, self.M, mse_hist = train_rfm(
            X_train,
            y_train,
            X_val,
            y_val,
            L=self.L,
            lam=self.lam,
            T=self.T,
            power=self.power,
            norm_control=norm_control,
            baseline=baseline,
        )
        return mse_hist

    def predict(self, X_test):
        return utils.K_M(X_test, self.X_train, self.M, self.L, self.power) @ self.alpha

    def score(self, X_test, y_test):
        return utils.mse(y_test, self.predict(X_test))


def train_rfm(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    power=1,
    L=1.0,
    lam=1e-3,
    T=10,
    norm_control=False,
    baseline=False,
):
    """
    Train an RFM kernel.

    Parameters
    ----------
    X_train: (n,d) numpy ndarray, training data
    y_train: (n,m) numpy ndarray, true outputs
    X_val: (w, d) numpy ndarray, validation data
    y_val: (w, m) numpy ndarray, true validation outputs
    L: float, bandwidth of kernel
    lam: float, regularization coefficient ("lambda")
    T: int, number of training iterations
    power: int, power of M matrix
    norm_control: bool (default False), whether to apply rsvd-based reconstruction during gradient calculation
    baseline: bool (default False), determines whether to use a 0th iteration kernel (don't run convergence) for baseline calculations


    Returns
    -------
    alpha: (n,1) numpy ndarray, contains weights for each training datapoint
    M: (d,d) numpy ndarray, contains trained weights of each feature
    """
    n, d = X_train.shape
    n, m = y_train.shape

    if X_val is None:
        X_val = X_train
        y_val = y_train

    val_mse_hist = []
    best_M = None
    best_alpha = None

    M = np.eye(d)
    if not baseline:
        for t in range(1, T + 1):
            K_train = utils.K_M(X_train, X_train, M, L, power)
            alpha = np.linalg.solve(K_train + lam * np.eye(n), y_train)
            M = utils.grad_laplace_mat(
                X_train, alpha, L=L, P=M, power=power, norm_control=norm_control
            )

            # evaluate on val, if this is the best so far, save it
            y_hat = utils.K_M(X_val, X_train, M, L, power) @ alpha
            val_mse = utils.mse(y_val, y_hat)
            if not val_mse_hist or val_mse < min(val_mse_hist):
                best_M = M
                best_alpha = alpha
            val_mse_hist.append(val_mse)

        logger.debug(
            "BEST VAL MSE: %.3f @ t=%d" % (min(val_mse_hist), np.argmin(val_mse_hist) + 1)
        )
    else:
        best_M = M
        K_train = utils.K_M(X_train, X_train, M, L, power)
        best_alpha = np.linalg.solve(K_train + lam * np.eye(n), y_train)

    # evaluate
    y_hat = (
        utils.K_M(X_train, X_train, best_M, L, power) @ best_alpha
    )  # (n,n) * (n, m) -> (n,m)

    logger.debug("TRAIN MSE: %.3f" % utils.mse(y_train, y_hat))

    return best_alpha, best_M, val_mse_hist


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
