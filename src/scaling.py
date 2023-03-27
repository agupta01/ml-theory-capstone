import logging
import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import trange

from rfm import RFM

# set dpi for plots
plt.rcParams["figure.dpi"] = 300

# train test split
split_size = 0.2

n, D = 1000, 2000

# setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] \t %(message)s",
    datefmt="%b %d %Y %I:%M%p",
)


# target function
f = lambda X: (
    5 * torch.pow(X[:, 0], 3)
    + 2 * torch.pow(X[:, 1], 2)
    + 10 * X[:, 2]
    # + torch.normal(mean=torch.zeros(X.shape[0]), std=torch.ones(X.shape[0])).cuda()
).reshape(-1, 1)

sinf = lambda X: (X[:, 2] * torch.sin(X[:, 0] * X[:, 1])).reshape(-1, 1)

true_K = torch.randn((10, 1), requires_grad=False, device="cuda")
randmat = lambda X: torch.matmul(X[:, :10], true_K[: X.shape[1], :]).reshape(-1, 1)

TARGET_FNS = {
    "cubic": f,
    "sinf": sinf,
    "randmat": randmat,
}


def run_one_sim(
    N=1000, noise=0.0, target_fn="cubic", norm_control=False, baseline=False
):
    train_MSE = []
    test_MSE = []
    mse_hist = []

    D = 2 * N
    X_full = torch.normal(
        mean=0.0, std=1.0, size=(N, D), requires_grad=False, device="cuda"
    )
    epsilon = torch.normal(
        mean=0.0, std=noise, size=(N,), requires_grad=False, device="cuda"
    )

    d_range = np.concatenate(
        (np.arange(5, int(0.1 * N), 1), np.arange(int(0.1 * N), D + 1, 10))
    )

    for d in d_range:
        X = X_full[:, :d] * (1 / np.sqrt(d))

        test_split_size = 0.2
        n_test_split = int(N * test_split_size)

        X_train, X_test = (
            X[: N - n_test_split],
            X[N - n_test_split :],
        )

        # recompute y
        y_train = TARGET_FNS[target_fn](X_train) + epsilon[: N - n_test_split].reshape(
            -1, 1
        )
        y_test = TARGET_FNS[target_fn](X_test) + epsilon[N - n_test_split :].reshape(
            -1, 1
        )

        model = RFM(
            backend="gpu", norm_control=norm_control, T=10 if not baseline else 0
        )
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


def run_sim(name="", N_runs=10, noise=0.0, N=1000, target_fn="cubic", baseline=False):
    train_MSEs = []
    test_MSEs = []

    logger.info(
        f"Running with parameters: {name}, {N_runs}, {noise}, {N}, {target_fn}, {baseline}"
    )

    for i in trange(N_runs):
        train_MSE, test_MSE = run_one_sim(
            N=N, noise=noise, target_fn=target_fn, baseline=baseline
        )
        train_MSEs.append(train_MSE)
        test_MSEs.append(test_MSE)

    train_MSEs = np.array(train_MSEs)
    test_MSEs = np.array(test_MSEs)

    # save results
    logging.info("Saving Results...")
    if name == "":
        name = f"{N_runs}runs_{N}N_{noise}noise_{target_fn}"
    baseline_str = "_baseline" if baseline else ""

    # change current dir if needed
    if os.getcwd().endswith("src"):
        os.chdir("..")
    # check if results folder exists, if not, create it
    if not os.path.exists("./results/arrays/scaling_results"):
        os.makedirs("./results/arrays/scaling_results")

    np.save(
        f"./results/arrays/scaling_results/train_MSEs_{name}{baseline_str}.npy",
        train_MSEs,
    )
    np.save(
        f"./results/arrays/scaling_results/test_MSEs_{name}{baseline_str}.npy",
        test_MSEs,
    )

    return


def generate_plots(train_MSEs, test_MSEs):
    # flatten all the observations to get each d's average
    train_MSE_mean = train_MSEs.mean(axis=0)
    test_MSE_mean = test_MSEs.mean(axis=0)

    # plot of train and test MSEs
    d_range = list(range(10, D + 1, 10))
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


# if __name__ == "__main__":
#     run_one_sim(N=200, noise=0.1, target_fn="cubic", norm_control=False, baseline=False)

if __name__ == "__main__":
    fire.Fire(run_sim)

    logging.info("Done.")
