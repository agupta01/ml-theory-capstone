import numpy as np
import src.utils as utils

TARGET_FNS = {
    "xsinx": utils.target_xsinx
}

def generate_test_data(target_fn_id: str, n_samples: int, noise_std: float, save_path: str = None):
    """Generates test dataset following the given target function with some noise.
    
    Parameters
    ----------
    target_fn_id : str
        Name of the target function to use.
    n_samples : int
        Number of samples to generate.
    noise_std : float   
        Standard deviation of the noise to add to the target function.
    save : bool, optional
        Whether to save the generated data to a file, by default False.
    
    Returns
    -------
    np.ndarray
        Generated data of shape (n_samples, 2).
    """
    x = np.random.rand(n_samples, 1) * 10 - 5
    y = TARGET_FNS[target_fn_id](x) + np.random.randn(*x.shape) * noise_std

    data = np.hstack([x, y])
    if save_path:
        np.savetxt(save_path, data, delimiter=",")
    return data