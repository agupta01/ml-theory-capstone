import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange

from src.utils import K_M, mse, K_laplace_mat
from src.rfm import RFM

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
f = lambda X: (
    5 * np.power(X[:, 0], 3)
    + 2 * np.power(X[:, 1], 2)
    + 10 * X[:, 2]
    + np.random.normal(size=X.shape[0], scale=0.1)
).reshape(-1, 1)


def run_one_sim(norm_control=True, baseline=False):
    train_MSE = []
    test_MSE = []
    mse_hist = []

    n, D = 1000, 500

    X_full = np.random.normal(size=(n, D))
    y = f(X_full)

    for d in range(10, 501):
        X = X_full[:, :d] * (1 / np.sqrt(d))

        test_split_size = 0.2
        n_test_split = int(n * test_split_size)

        X_train, X_test = (
            X[: n - n_test_split],
            X[n - n_test_split :],
        )

        # recompute y
        y_train = f(X_train)
        y_test = f(X_test)

        model = RFM(backend="gpu", norm_control=norm_control, T=10 if not baseline else 0)
        mse_hist.append(
            model.fit(
                X_train,
                y_train,
                val_split=0.2,
            )
        )
        train_MSE.append(model.score(X_train, y_train))
        test_MSE.append(model.score(X_test, y_test))

    return np.array(train_MSE), np.array(test_MSE)


def run_sim(N_runs=10, norm_control=True, baseline=False, plot=True):
    train_MSEs = []
    test_MSEs = []

    for i in trange(N_runs):
        train_MSE, test_MSE = run_one_sim(norm_control, baseline)
        train_MSEs.append(train_MSE)
        test_MSEs.append(test_MSE)

    train_MSEs = np.array(train_MSEs)
    test_MSEs = np.array(test_MSEs)

    if plot:
        generate_plots(train_MSEs, test_MSEs)

    return (train_MSEs, test_MSEs)


def generate_plots(train_MSEs, test_MSEs):
    # flatten all the observations to get each d's average
    train_MSE_mean = train_MSEs.mean(axis=0)
    test_MSE_mean = test_MSEs.mean(axis=0)

    # plot of train and test MSEs
    d_range = list(range(10, 501))
    d_range_exploded = np.repeat(d_range, train_MSEs.shape[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle("Train MSE vs Test MSE")

    ax1.set_xlabel("d")
    ax1.scatter(
        d_range_exploded,
        train_MSEs.flatten(order="F"),
        color="blue",
        alpha=0.1,
        s=5,
        label="Train MSE",
    )
    ax1.plot(d_range, train_MSE_mean, color="blue", label="Train MSE", linewidth=2)
    ax1.set_yscale("log")
    ax1.set_ylabel("Train MSE")

    ax2.set_xlabel("d")
    ax2.scatter(
        d_range_exploded,
        test_MSEs.flatten(order="F"),
        color="r",
        alpha=0.1,
        s=5,
        label="Test MSE",
        linewidth=2,
    )
    ax2.plot(d_range, test_MSE_mean, color="r", label="Test MSE")
    ax2.set_yscale("log")
    # ax2.set_xscale("log")
    ax2.set_ylabel("Test MSE")

    plt.savefig("./results/plots/train_test_MSE.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N_runs", type=int, default=10)
    parser.add_argument("--norm_control", type=str, default="true")
    parser.add_argument("--plot", type=bool, default=True)
    args = parser.parse_args()

    args.norm_control = args.norm_control.lower() == "true"

    logging.info(f"Running {args.N_runs} runs")
    logging.info(f"norm_control: {args.norm_control}")

    train_MSEs, test_MSEs, M_norms, stdevs = run_sim(
        args.N_runs, args.norm_control, args.plot
    )

    used_M_norm = "_norm_control" if args.norm_control else ""

    np.save(f"./results/arrays/train_MSEs{used_M_norm}_{str(args.N_runs)}.npy", train_MSEs)
    np.save(f"./results/arrays/test_MSEs{used_M_norm}_{str(args.N_runs)}.npy", test_MSEs)

    logging.info("Done.")
