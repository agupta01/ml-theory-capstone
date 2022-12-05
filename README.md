# Benchmarking Kernel Machines on Image Datasets (Working Title)
Data Science Capstone Project advised by Mikhail Belkin.

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