import numpy as np
import matplotlib.pyplot as plt
import src.utils as utils
from src.etl import TARGET_FNS



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
        print("MSE:", np.mean((yhat - y)**2))

    if save_fig_path:
        plt.plot(X, Y, 'o', label="data")
        plt.plot(domain, yhat, label="prediction")
        plt.legend()
        plt.savefig(save_fig_path)
    
    return yhat