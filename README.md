# Benchmarking Kernel Machines on Text Datasets
Data Science Capstone Project advised by Mikhail Belkin.

## Scaling Tests
If you would like to run the expriments used in the report "On Feature Scaling of Recursive Feature Machines", follow the steps below:

1. Have a GPU and a GPU-enabled Pytorch v1.13 environment (feel free to use the environment.yml file in the repo, although this contains a lot of other stuff).
2. Navigate to `src` within the project root (`cd src`)
3. Run the following:
```shell
python scaling.py --name=<PROVIDE A NAME HERE> --noise=<specify noise to add to dataset here> --N_runs=<set to 100 for 100 trials> --N=<number of examples in dataset> --target_fn=<cubic for the default function, randmat for the random matrix function> --baseline=<True to run baseline (Laplacian) kernel, False to train full RFM)
```
4. After the experiment is run (100 trials takes about 30 minutes when N=1000 on a RTX 2060), you can find result files in `<project root>/results/arrays/scaling_results`. Every run will generate two numpy arrays, named `train_MSEs_<name>.npy` and `test_MSEs_<name>.npy`, appended with "\_baseline" if a baseline run was used. Each array has shape (N_runs, len(d_range)), where d_range are the feature sizes attempted ([5, 6, 7, ..., 99] + [100, 110, 120, ..., 2000] in the base experiment in the original paper).

## How to use
```shell
usage: run.py [-h] [--verbose] {test,test-data,mnist,cifar10,fashionmnist}

positional arguments:
  {test,test-data,mnist,cifar10,fashionmnist}
                        task to run

optional arguments:
  -h, --help            show this help message and exit
  --verbose             Set logging to DEBUG
```

Example:

```shell
$ python run.py mnist --verbose
Dec 04 2022 11:38PM [DEBUG] 	 Logging set to DEBUG
Dec 04 2022 11:38PM [DEBUG] 	 Using 1259 training samples and 210 test samples
Dec 04 2022 11:38PM [DEBUG] 	 x_train shape: (1259, 784)
Dec 04 2022 11:38PM [DEBUG] 	 y_train shape: (1259,)
Dec 04 2022 11:39PM [INFO] 	 TRAIN MSE: 0.000 | Accuracy: 1.000
 TEST MSE: 0.019 | Accuracy: 0.995 | Precision: 0.995 | Recall: 0.992
```

The corresponding config files for each dataset can be found in `config/<dataset>.json`. Below is an explanation of the options:

```json
{
    "kernel_type": "laplace", # can be either laplace or gaussian
    "print_result": true, # to print result to log
    "data": {
        "dataset": "mnist", # dataset to use, can be "mnist", "cifar10", or "fashionmnist"
        "subset": 0.1, # subset of dataset to use. See note below.
        "pos_class": 1, # class label for positive class (1)
        "neg_class": 8 # class label for negative class (-1)
    },
    "model": {
        "gamma": 0.00128, # kernel bandwidth, see src/utils.py for further details
        "return_metrics": true # return metrics rather than predictions after training kernel
    }
}
```

## A note on not breaking your computer
This code produces pairwise distance kernels for use in kernel machines for binary classification. If your dataset
has $n$ examples, your kernel will be $n \times n$! My 2018 Macbook Pro with 16GB RAM can only handle $n \approx 1000$ 
before it starts to freeze up on itself. Use the `subset` parameter in the config file, do the math, and you may 
avoid bricking your laptop for 5 minutes.

If you're using DSMLP, subset = 0.01 should work for most datasets.
