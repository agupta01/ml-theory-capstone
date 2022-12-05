import logging
import numpy as np
import matplotlib.pyplot as plt
import src.utils as utils
from src.etl import TARGET_FNS
from functools import partial

logger = logging.getLogger(__name__)

def poly_regression(data, lam: float, gamma: float, p: float, func_id: str = None, save_fig_path: str = None):
    """Run polynomial kernel regression with the specified parameters.
    
    Parameters
    ----------
    data : np.ndarray
        Data to fit the model to.
    lam : float
        Regularization parameter.
    gamma : float
        Gamma parameter for the polynomial kernel.
    p : float
        P parameter for the polynomial kernel.
    func_id : str, optional
        True data generating function
    save_fig_path : str, optional
        Path to save the plot of the results, by default None.
    
    Returns
    -------
    np.ndarray
        Predictions of the model on the data.
    """
    domain = np.linspace(-5, 5, 1000).reshape(-1, 1)
    X, Y = data[:, 0].reshape(-1, 1), data[:, 1]
    K = utils.K_poly_mat(X, X, gamma, p)
    alpha_hat = np.linalg.solve(K+lam*np.eye(len(X)), Y)
    yhat = utils.K_poly_mat(domain, X, gamma, p) @ alpha_hat

    if func_id is not None:
        y = TARGET_FNS[func_id](X)
        logger.info("MSE: %.3f", np.mean((yhat - y)**2))

    if save_fig_path:
        plt.plot(X, Y, 'o', label="data")
        plt.plot(domain, yhat, label="prediction")
        plt.legend()
        plt.savefig(save_fig_path)
    
    return yhat

def classify(type: str, train_data, test_data, return_metrics: bool = True, **kwargs):
    """Perform classification.
    
    Parameters
    ----------
    type : str
        Type of kernel to use.
    data : np.ndarray
        Data to fit the model to.
    return_metrics : bool, optional
        Whether to return the accuracy of the model, by default True.
    **kwargs
        Additional parameters for the classification function.
    """
    if type == "laplace":
        kernel = partial(utils.K_laplace_mat, gamma=kwargs["gamma"])
    elif type == "gaussian":
        kernel = partial(utils.K_gauss_mat, gamma=kwargs["gamma"])
    else:
        raise ValueError(f"Unknown kernel type: {type}")

    lam = 0.1
    (x_train, y_train) = train_data
    (x_test, y_test) = test_data
    K = kernel(x_train, x_train)
    alpha_hat = np.linalg.solve(K + lam*np.eye(len(x_train)), y_train)
    yhat = np.sign((kernel(x_train, x_train) @ alpha_hat) + 1e-12) # add a small epsilon to avoid zeros
    yhat_test = np.sign((kernel(x_test, x_train) @ alpha_hat) + 1e-12)
    tp = np.sum(np.logical_and(yhat_test == 1, y_test == 1))
    fp = np.sum(np.logical_and(yhat_test == 1, y_test == -1))
    fn = np.sum(np.logical_and(yhat_test == -1, y_test == 1))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if return_metrics:
        result = "TRAIN MSE: %.3f" % np.mean((yhat - y_train)**2) + " | Accuracy: %.3f" % np.mean(yhat == y_train)
        result += "\n TEST MSE: %.3f" % np.mean((yhat_test - y_test)**2) + " | Accuracy: %.3f" % np.mean(yhat_test == y_test) + " | Precision: %.3f" % precision + " | Recall: %.3f" % recall
        return result
    return yhat