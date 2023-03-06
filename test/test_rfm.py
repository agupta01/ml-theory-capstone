"""
Tests for RFM model
"""
import numpy as np
from numpy.random import default_rng
import torch
from src import gpu_utils, utils
from src.rfm import RFM


def test_create_rfm():
    """
    Test that we can create an RFM object
    """
    rfm = RFM()
    assert rfm is not None


def test_single_rfm():
    """
    Test that a basic RFM trains and gives predictable results.
    """
    rng = default_rng(seed=42)
    # set up data
    X = rng.normal(size=(100, 10))
    y = (X[:, 0] + X[:, 1] + X[:, 2]).reshape(-1, 1)

    # set up model
    rfm = RFM()

    # train model
    rfm.fit(X, y)

    # predict
    y_hat = rfm.predict(X)

    # check that the shapes are correct
    assert y_hat.shape == y.shape
    # mse should be 0.0018631536196526705 with seed=42
    assert np.isclose(rfm.score(X, y), 0.0018631536196526705)


def test_backends_equal():
    """
    Train two RFMs with different backends on the same data and check that the results are the same.
    """
    rng = default_rng(seed=42)
    # set up data
    X = rng.normal(size=(100, 10))
    y = (X[:, 0] + X[:, 1] + X[:, 2]).reshape(-1, 1)

    # set up models
    rfm = RFM()
    rfm2 = RFM(backend="opt")

    # train models
    rfm.fit(X, y)
    rfm2.fit(X, y)

    # predict
    y_hat = rfm.predict(X)
    y_hat2 = rfm2.predict(X)

    # check that the shapes are correct
    assert y_hat.shape == y.shape
    assert y_hat2.shape == y.shape

    # check that the predictions are the same
    assert np.allclose(y_hat, y_hat2)


def test_gpu():
    """
    Test that we can use the GPU backend.
    """
    rng = default_rng(seed=42)
    # set up data
    X = rng.normal(size=(100, 10))
    y = (X[:, 0] + X[:, 1] + X[:, 2]).reshape(-1, 1)

    # set up model
    rfm = RFM(backend="gpu")

    # train model
    rfm.fit(X, y)

    # predict
    y_hat = rfm.predict(X)

    # check that the shapes are correct
    assert y_hat.shape == y.shape

def test_gpu_matches_cpu():
    """
    Test that the GPU backend gives the same results as the CPU backend.
    """
    rng = default_rng(seed=42)
    # set up data
    X = rng.normal(size=(100, 10))
    y = (X[:, 0] + X[:, 1] + X[:, 2]).reshape(-1, 1)

    # set up models
    rfm = RFM()
    rfm2 = RFM(backend="gpu")

    # train models
    rfm.fit(X, y)
    rfm2.fit(X, y)

    # predict
    y_hat = rfm.predict(X)
    y_hat2 = rfm2.predict(X).cpu().numpy()

    # check that the shapes are correct
    assert y_hat.shape == y.shape
    assert y_hat2.shape == y.shape

    # check that the predictions are the same
    try:
        assert np.allclose(y_hat, y_hat2, atol=1e-4)
    except AssertionError:
        print("||yhat - yhat2|| =", np.linalg.norm(y_hat - y_hat2))
        assert False


def test_grad_laplace_util_matches():
    """
    Test that utils.grad_laplace_mat and gpu_utils.grad_laplace_mat_gpu give the same results.
    """
    rng = default_rng(seed=42)
    # set up data
    n, d = 100, 5
    X = rng.normal(size=(n, d))
    alpha = rng.normal(size=(n, 1))
    L = 1.0
    M = np.eye(d)

    # convert to torch
    X_gpu = torch.from_numpy(X).float()
    alpha_gpu = torch.from_numpy(alpha).float()
    M_gpu = torch.from_numpy(M).float()

    # check K_M
    # K_M_cpu = utils.K_M(X, X, M, L)
    # K_M_gpu = gpu_utils.K_M(X_gpu, X_gpu, M_gpu, L).cpu().numpy()

    # # check that the shapes are correct
    # assert K_M_cpu.shape == K_M_gpu.shape

    # # check that the values are the same
    # try:
    #     assert np.allclose(K_M_cpu, K_M_gpu)
    # except AssertionError:
    #     print("||K_M_cpu - K_M_gpu|| =", np.linalg.norm(K_M_cpu - K_M_gpu))
    #     assert False

    # # check K_M_grad
    # K_M_grad_cpu = utils.K_M_grad(X, X, M, L)
    # K_M_grad_gpu = gpu_utils.K_M_grad(X_gpu, X_gpu, M_gpu, L).cpu().numpy()

    # # check that the shapes are correct
    # assert K_M_grad_cpu.shape == K_M_grad_gpu.shape

    # # check that the values are the same
    # try:
    #     assert np.allclose(K_M_grad_cpu, K_M_grad_gpu)
    # except AssertionError:
    #     print("||K_M_grad_cpu - K_M_grad_gpu|| =", np.linalg.norm(K_M_grad_cpu - K_M_grad_gpu))
    #     assert False

    # check outputs of both functions
    M_cpu = utils.grad_laplace_mat(X, alpha, L, M)
    M_gpu = gpu_utils.grad_laplace_mat_gpu(X_gpu, alpha_gpu, L, M_gpu).cpu().numpy()

    # check that the shapes are correct
    assert M_cpu.shape == M_gpu.shape

    # check that the values are the same
    try:
        assert np.allclose(M_cpu, M_gpu, atol=1e-4)
    except AssertionError:
        print("||M_cpu - M_gpu|| =", np.linalg.norm(M_cpu - M_gpu))
        assert False