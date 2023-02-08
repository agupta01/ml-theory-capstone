import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange

from utils import K_M
from utils import grad_laplace_mat as gm

# grad override
grad_laplace_mat = lambda a, x, z, M, L: gm(x, a, L, M)

# set dpi for plots
plt.rcParams["figure.dpi"] = 300

# regularization parameter
_lambda = 1e-3

# train test split
split_size = 0.2

# setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] \t %(message)s",
    datefmt="%b %d %Y %I:%M%p",
)


# target function
f = lambda x: (5*x[0]**3) + (2*x[1]**2) + (10*x[2])


def train_for_scaling(X_train, y_train, L=1.0, T=10):
    n, d = X_train.shape
    M = np.eye(d)
    for t in range(T):
        K_train = K_M(X_train, X_train, M, L=L)
        alpha = y_train @ np.linalg.pinv(K_train + _lambda * np.eye(n))
        M = grad_laplace_mat(alpha, X_train, X_train, M, L=L)

    return alpha, M


def apply_norm_control(M, top_k=0.1, strategy="eig", debug=False):
    """Use top-k eigenvalues or SVD to control M's norm"""
    # take the top 10% of the significant values
    k = int(M.shape[0] * top_k)
    if strategy == "eig":
        w, v = np.linalg.eigh(M)
        w, v = w[k:], v[:, k:]

        # reconstruct M
        N = np.dot(v, np.dot(np.diag(w), v.T))
    elif strategy == "svd":
        u, s, v = np.linalg.svd(M, full_matrices=False)
        N = np.dot(u[:, :k], np.dot(np.diag(s[:k]), v[:k, :]))
    else:
        raise ValueError("Strategy must be 'eig' or 'svd'.")

    if debug:
        print(f"[Norm Control] starting norm: {np.linalg.norm(M)}")
        print(
            f"[Norm Control] ending norm: {np.linalg.norm(N)} \t distance: {np.linalg.norm(M - N)}"
        )

    return N


def run_one_sim(stdev_control=True, norm_control=True):
    train_MSE = []
    test_MSE = []
    M_norm = []
    stdevs = []

    L = 1.

    n, _d = (100, 200)  # starting size

    # Regenerate training data
    X_train = np.random.normal(size=(int(n * (1 - split_size)), _d))
    X_test = np.random.normal(size=(int(n * split_size), _d))

    for d in range(10, 200):
        # slice and renormalize data
        train_X = X_train[:, :d] * (1 / np.sqrt(d))
        test_X = X_test[:, :d] * (1 / np.sqrt(d))

        if stdev_control:
            train_X *= 1 / np.sqrt(d)
            test_X *= 1 / np.sqrt(d)

        stdevs.append(np.std(train_X))

        # recompute y_train and y_test
        train_y = np.array([f(x) for x in train_X])
        test_y = np.array([f(x) for x in test_X])

        alpha, M = train_for_scaling(train_X, train_y)

        if norm_control:
            M = apply_norm_control(M, strategy="eig")

        M_norm.append(np.linalg.norm(M))

        # train mse
        y_pred_train = alpha @ K_M(train_X, train_X, M, L)
        train_MSE.append(((train_y - y_pred_train) ** 2).mean())

        # test mse
        y_pred_test = alpha @ K_M(train_X, test_X, M, L)
        test_MSE.append(((test_y - y_pred_test) ** 2).mean())

    return np.array(train_MSE), np.array(test_MSE), np.array(M_norm), np.array(stdevs)


def run_sim(N_runs=10, stdev_control=True, norm_control=True, plot=True):
    train_MSEs = []
    test_MSEs = []
    M_norms = []
    stdevs = []

    for i in trange(N_runs):
        train_MSE, test_MSE, M_norm, stdev = run_one_sim(stdev_control, norm_control)
        train_MSEs.append(train_MSE)
        test_MSEs.append(test_MSE)
        M_norms.append(M_norm)
        stdevs.append(stdev)

    train_MSEs = np.array(train_MSEs)
    test_MSEs = np.array(test_MSEs)
    M_norms = np.array(M_norms)
    stdevs = np.array(stdevs)

    if plot:
        generate_plots(train_MSEs, test_MSEs, M_norms, stdevs)

    return (
        train_MSEs,
        test_MSEs,
        M_norms,
        stdevs,
    )

def generate_plots(train_MSEs, test_MSEs, M_norms, stdevs):
    # flatten all the observations to get each d's average
    train_MSE_mean = train_MSEs.mean(axis=0)
    test_MSE_mean = test_MSEs.mean(axis=0)



    # plot of train and test MSEs
    d_range = list(range(10, 200))
    d_range_exploded = np.repeat(d_range, train_MSEs.shape[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle("Train MSE vs Test MSE")

    ax1.set_xlabel("d")
    ax1.scatter(d_range_exploded, train_MSEs.flatten(order='F'), color='blue', alpha=0.1, s=5, label="Train MSE")
    ax1.plot(d_range, train_MSE_mean, color='blue', label="Train MSE", linewidth=2)
    ax1.set_yscale("log")
    ax1.set_ylabel("Train MSE")

    ax2.set_xlabel("d")
    ax2.scatter(d_range_exploded, test_MSEs.flatten(order='F'), color='r', alpha=0.1, s=5, label="Test MSE", linewidth=2)
    ax2.plot(d_range, test_MSE_mean, color='r', label="Test MSE")
    ax2.set_yscale("log")
    # ax2.set_xscale("log")
    ax2.set_ylabel("Test MSE")

    plt.savefig("../results/plots/train_test_MSE.png")

    # plot M_norm vs test_MSE
    fig = plt.figure(figsize=(9, 9))
    sns.regplot(x=test_MSEs.flatten(), y=M_norms.flatten())

    plt.xlabel("Test MSE")
    plt.ylabel("||M||")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("../results/plots/M_norm_vs_test_MSE.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_runs", type=int, default=10)
    parser.add_argument("--stdev_control", type=str, default="true")
    parser.add_argument("--norm_control", type=str, default="true")
    parser.add_argument("--plot", type=bool, default=True)
    args = parser.parse_args()

    args.stdev_control = args.stdev_control.lower() == "true"
    args.norm_control = args.norm_control.lower() == "true"

    logging.info(f"Running {args.N_runs} runs")
    logging.info(f"stdev_control: {args.stdev_control}")
    logging.info(f"norm_control: {args.norm_control}")

    train_MSEs, test_MSEs, M_norms, stdevs = run_sim(
        args.N_runs, args.stdev_control, args.norm_control, args.plot
    )

    used_M_norm = "_norm_control" if args.norm_control else ""
    used_stdev = "_stdev_control" if args.stdev_control else ""

    np.save(f"../results/arrays/train_MSEs{used_M_norm}{used_stdev}_eig.npy", train_MSEs)
    np.save(f"../results/arrays/test_MSEs{used_M_norm}{used_stdev}_eig.npy", test_MSEs)
    np.save(f"../results/arrays/M_norms{used_M_norm}{used_stdev}_eig.npy", M_norms)
    np.save(f"../results/arrays/stdevs{used_M_norm}{used_stdev}_eig.npy", stdevs)

    logging.info("Done.")
