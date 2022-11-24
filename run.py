import argparse
import json
from src.main import poly_regression
from src.etl import generate_test_data

parser = argparse.ArgumentParser()

# add required argument
parser.add_argument("task", help="task to run", choices=["test", "test-data"])

# parse arguments
args = parser.parse_args()

# run task
if args.task == "test":
    # first, generage test data
    test_data_config = json.load(open("config/test_data.json"))
    del test_data_config["save_path"]

    test_data = generate_test_data(**test_data_config)

    poly_regression_config = json.load(open("config/poly_regression.json"))
    poly_regression(test_data, **poly_regression_config)
elif args.task == "test-data":
    # build test data
    config = json.load(open("config/test_data.json"))
    generate_test_data(**config)