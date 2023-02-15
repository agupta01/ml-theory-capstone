import argparse
import json
import logging
import numpy as np
from src.utils import K_M, mse
from src.main import classify, poly_regression
from src.rfm import train_rfm, test_rfm, plot_results
from src.etl import generate_test_data, load_data
from src.scaling import run_sim

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] \t %(message)s",
    datefmt="%b %d %Y %I:%M%p",
)

# add required argument
parser.add_argument(
    "task",
    help="task to run",
    choices=[
        "test",
        "test-data",
        "mnist",
        "cifar10",
        "fashionmnist",
        "rfm",
        "rfmpowertest",
        "scaling",
    ],
)
parser.add_argument("--verbose", help="Set logging to DEBUG", action="store_true")
# scaling args
parser.add_argument("--N_runs", type=int, default=10)
parser.add_argument("--norm_control", type=str, default="true")
parser.add_argument("--plot", type=bool, default=True)

# parse arguments
args = parser.parse_args()

if args.verbose:
    logger.setLevel(logging.DEBUG)
    logger.debug("Logging set to DEBUG")

# run task
if args.task == "test":
    # first, generate test data: polynomial regression with noise
    test_data_config = json.load(open("config/test_data.json"))
    del test_data_config["save_path"]

    test_data = generate_test_data(**test_data_config)

    poly_regression_config = json.load(open("config/poly_regression.json"))
    poly_regression(test_data, **poly_regression_config)

elif args.task == "test-data":
    # build test data: polynomial regression with noise
    config = json.load(open("config/test_data.json"))
    generate_test_data(**config)

elif args.task == "mnist" or args.task == "cifar10" or args.task == "fashionmnist":
    # load mnist data
    config = json.load(open(f"config/{args.task}.json"))
    train_data, test_data = load_data(logger=logger, **config["data"])

    # perform regression according to specified parameters
    result = classify(config["kernel_type"], train_data, test_data, **config["model"])

    if config["print_result"]:
        logger.info(result)

elif args.task == "rfm":
    # generate data
    train_data = generate_test_data(target_fn_id="xsinx", n_samples=1000, noise_std=0.1)
    test_data = generate_test_data(target_fn_id="xsinx", n_samples=200, noise_std=0.1)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # load config
    config = json.load(open(f"config/{args.task}_naive.json"))
    L = config["L"]

    # train and test model
    alpha, M = train_rfm(X_train, y_train, **config)
    test_rfm(X_train, X_test, y_test, alpha, M, L)

    # plot results
    y_hat = alpha @ K_M(X_train, X_train, M, L)
    plot_results(X_train, y_train, y_hat, "results/plots/trainrfm.png")

    y_hat = alpha @ K_M(X_train, X_test, M, L)
    plot_results(X_test, y_test, y_hat, "results/plots/testrfm.png", test=True)

elif args.task == "rfmpowertest":
    # generate data
    train_data = generate_test_data(target_fn_id="xsinx", n_samples=1000, noise_std=0.1)
    test_data = generate_test_data(target_fn_id="xsinx", n_samples=200, noise_std=0.1)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # load config
    config = json.load(open(f"config/rfm_naive.json"))
    L = config["L"]

    # increase power of M successively
    for i in range(1, 4):
        logging.info(f"M Power Test @ {i}")
        alpha, M = train_rfm(X_train, y_train, power=i, **config)
        test_rfm(X_train, X_test, y_test, alpha, M, L, power=i)

elif args.task == "scaling":
    args.norm_control = args.norm_control.lower() == "true"

    logging.info(f"Running {args.N_runs} runs")
    logging.info(f"norm_control: {args.norm_control}")

    train_MSEs, test_MSEs = run_sim(args.N_runs, args.norm_control, args.plot)

    used_M_norm = "_norm_control" if args.norm_control else ""

    np.save(f"./results/arrays/train_MSEs{used_M_norm}.npy", train_MSEs)
    np.save(f"./results/arrays/test_MSEs{used_M_norm}.npy", test_MSEs)

    logging.info("Done.")
