import argparse
import json
import logging
from src.main import classify, poly_regression
from src.etl import generate_test_data, load_data

parser = argparse.ArgumentParser()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] \t %(message)s", datefmt="%b %d %Y %I:%M%p")

# add required argument
parser.add_argument("task", help="task to run", choices=["test", "test-data", "mnist", "cifar10", "fashionmnist"])
parser.add_argument("--verbose", help="Set logging to DEBUG", action="store_true")

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
    result = classify(config['kernel_type'], train_data, test_data, **config["model"])

    if config["print_result"]:
        logger.info(result)