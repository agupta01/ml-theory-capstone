"""
Tests for RFM model
"""
import numpy as np
from numpy.random import default_rng
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