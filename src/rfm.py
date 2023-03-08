import logging
from typing import Union
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from . import utils
from . import gpu_utils
from .etl import TARGET_FNS

logger = logging.getLogger(__name__)


class RFM:
    def __init__(
        self, backend="naive", norm_control=False, L=1.0, lam=1e-3, T=10, power=1
    ):
        self.backend = backend
        self.norm_control = norm_control
        self.L = L
        self.lam = lam
        self.T = T
        self.power = power
        self.baseline = False  # deprecated; set T = 0 for baseline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y, val_split=0.0):
        # split into train and val, if specified
        if val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split)
        else:
            X_train = X
            y_train = y
            X_val = None
            y_val = None

        if self.backend == "gpu":
            # convert to torch if necessary
            if not isinstance(X_train, torch.Tensor):
                X_train = torch.from_numpy(X_train).to(self.device).float()
            if not isinstance(y_train, torch.Tensor):
                y_train = torch.from_numpy(y_train).to(self.device).float()
            if X_val is not None:
                if not isinstance(X_val, torch.Tensor):
                    X_val = torch.from_numpy(X_val).to(self.device).float()
                if not isinstance(y_val, torch.Tensor):
                    y_val = torch.from_numpy(y_val).to(self.device).float()

        self.X_ = X_train

        if self.backend == "gpu":
            self.alpha_, self.M_, mse_hist = train_rfm_gpu(
                X_train,
                y_train,
                X_val,
                y_val,
                L=self.L,
                lam=self.lam,
                T=self.T,
                norm_control=self.norm_control,
                device=self.device,
            )
        else:
            self.alpha_, self.M_, mse_hist = train_rfm(
                X_train,
                y_train,
                X_val,
                y_val,
                L=self.L,
                lam=self.lam,
                T=self.T,
                power=self.power,
                norm_control=self.norm_control,
                baseline=self.baseline,
                backend=self.backend,
            )
        return mse_hist

    def predict(self, X):
        if self.backend == "gpu":
            # cnvert to torch tensor if necessary
            if not isinstance(X, torch.Tensor):
                X = torch.from_numpy(X).to(self.device).float()
            
            return gpu_utils.K_M(X, self.X_, self.M_, self.L) @ self.alpha_
        else:
            return utils.K_M(X, self.X_, self.M_, self.L, self.power) @ self.alpha_

    def score(self, X, y) -> float:
        if self.backend == 'gpu':
            return utils.mse(y.cpu().numpy(), self.predict(X).cpu().numpy())
        else:
            return utils.mse(y, self.predict(X))


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
    backend="naive",
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
    backend: str, determines which backend to use for training. Options are "naive" (basic dense numpy), "opt" (optimized dense numpy), and "gpu" (GPU-accelerated)

    Returns
    -------
    alpha: (n,1) numpy ndarray, contains weights for each training datapoint
    M: (d,d) numpy ndarray, contains trained weights of each feature
    """
    n, d = X_train.shape
    n, m = y_train.shape

    grad_func = {
        "naive": utils.grad_laplace_mat,
        "opt": utils.grad_laplace_mat_opt,
    }

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
            M = grad_func[backend](
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
            "BEST VAL MSE: %.3f @ t=%d"
            % (min(val_mse_hist), np.argmin(val_mse_hist) + 1)
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


def train_rfm_gpu(
    X_train: torch.TensorType,
    y_train: torch.TensorType,
    X_val: Union[torch.TensorType, None] = None,
    y_val: Union[torch.TensorType, None] = None,
    L: float = 1.0,
    lam: float = 1e-3,
    T: int = 4,
    norm_control: bool = False,
    device: str = "cuda",
):
    """
    Train an RFM kernel using GPU acceleration.

    Parameters
    ----------
    X_train: (n,d) torch tensor, training data
    y_train: (n,m) torch tensor, true outputs
    X_val: (w, d) torch tensor, validation data
    y_val: (w, m) torch tensor, true validation outputs
    L: float, bandwidth of kernel
    lam: float, regularization coefficient ("lambda")
    T: int, number of training iterations
    norm_control: bool (default False), whether to apply rsvd-based reconstruction during gradient calculation
    device: str, device to use for training. Options are "cuda" and "cpu"

    Returns
    -------
    alpha: (n,1) torch tensor, contains weights for each training datapoint
    M: (d,d) torch tensor, contains trained weights of each feature
    val_mse_hist: list of floats, contains validation MSE at each iteration
    """
    n, d = X_train.shape
    n, m = y_train.shape

    if X_val is None:
        X_val = X_train
        y_val = y_train

    val_mse_hist = []
    best_M = torch.eye(d, device=device)
    best_alpha = None

    M = torch.eye(d, device=device)
    reg = lam * torch.eye(n, device=device)
    for t in range(1, T + 1):
        K_train = gpu_utils.K_M(X_train, X_train, M, L)
        alpha = torch.linalg.solve(K_train + reg, y_train)
        M = gpu_utils.grad_laplace_mat_gpu(
            X=X_train, a=alpha, L=L, M=M, norm_control=norm_control
        )

        # evaluate on val, if this is the best so far, save it
        y_hat = gpu_utils.K_M(X_val, X_train, M, L) @ alpha
        val_mse = utils.mse(y_val, y_hat).cpu().item()
        if not val_mse_hist or val_mse < min(val_mse_hist):
            best_M = M
            best_alpha = alpha
        val_mse_hist.append(val_mse)

        logger.debug(
            "Iteration %d BEST VAL MSE: %.3f @ t=%d"
            % (t, min(val_mse_hist), np.argmin(val_mse_hist) + 1)
        )

    if best_alpha is None:
        # train a baseline (0th iteration kernel)
        K_train = gpu_utils.K_M(X_train, X_train, M, L)
        best_alpha = torch.linalg.solve(K_train + reg, y_train)

    # evaluate
    y_hat = (
        gpu_utils.K_M(X_train, X_train, best_M, L) @ best_alpha
    )  # (n,n) * (n, m) -> (n,m)

    logger.debug("TRAIN MSE: %.3f" % utils.mse(y_train, y_hat).cpu().item())

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
