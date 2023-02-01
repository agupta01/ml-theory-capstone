from logging import Logger

import numpy as np
import src.utils as utils
from sklearn.datasets import make_classification
import hashlib
import src.utils as utils
import os
import requests
import json
from bs4 import BeautifulSoup

TARGET_FNS = {
    "xsinx": utils.target_xsinx
}

def generate_test_data(target_fn_id: str, n_samples: int, noise_std: float, save_path: str = None):
    """Generates test dataset following the given target function with some noise.
    
    Parameters
    ----------
    target_fn_id : str
        Name of the target function to use.
    n_samples : int
        Number of samples to generate.
    noise_std : float   
        Standard deviation of the noise to add to the target function.
    save : bool, optional
        Whether to save the generated data to a file, by default False.
    
    Returns
    -------
    np.ndarray
        Generated data of shape (n_samples, 2).
    """
    x = np.random.rand(n_samples, 1) * 10 - 5
    y = TARGET_FNS[target_fn_id](x) + np.random.randn(*x.shape) * noise_std

    data = np.hstack([x, y])
    if save_path:
        np.savetxt(save_path, data, delimiter=",")
    return data

def generate_test_classification(n_samples: int, n_features: int, n_classes: int, save_path: str = None):
    """Generates a random classification dataset using sklearn make_classification.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Number of features to generate.
    n_classes : int
        Number of classes to generate.
    save_path : str, optional
        Path to save the generated data to, by default None.
    
    Returns
    -------
    np.ndarray
        Generated data.
    """
    x, y = make_classification(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_features, n_redundant=0, n_repeated=0, n_clusters_per_class=1, random_state=0)
    data = np.hstack([x, y.reshape(-1, 1)])
    if save_path:
        np.savetxt(save_path, data, delimiter=",")
    return data

def preproc_data(x_train, y_train, x_test, y_test, subset: float, pos_class: int, neg_class: int, logger: Logger):
    """Helper function to filter -> subset -> binarize data.
    
    Parameters
    ----------
    x_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Training labels.
    x_test : np.ndarray
        Test data.
    y_test : np.ndarray
        Test labels.
    subset : float
        Fraction of the dataset to use.
    pos_class : int
        Positive class to use.
    neg_class : int
        Negative class to use.
    logger : Logger
        Logger to use.

    Returns
    -------
    Preprocessed dataset.
    """
    # Reshape the data and rescale
    x_train = x_train.reshape(x_train.shape[0], -1).astype(float) / 255.0
    x_test = x_test.reshape(x_test.shape[0], -1).astype(float) / 255.0

    # Filter the data
    train_idxs = ((y_train == pos_class) | (y_train == neg_class)).flatten()
    test_idxs = ((y_test == pos_class) | (y_test == neg_class)).flatten()
    x_train = x_train[train_idxs]
    y_train = y_train[train_idxs]
    x_test = x_test[test_idxs]
    y_test = y_test[test_idxs]

    # Subset the data
    n_train = int(x_train.shape[0] * subset)
    n_test = int(x_test.shape[0] * subset)
    logger.debug("Using %d training samples and %d test samples", n_train, n_test)
    x_train = x_train[:n_train]
    y_train = y_train[:n_train]
    x_test = x_test[:n_test]
    y_test = y_test[:n_test]

    # Change to signed ints
    y_train = y_train.astype(np.int8)
    y_test = y_test.astype(np.int8)

    # Convert to binary labels
    y_train[y_train == pos_class] = 1
    y_train[y_train == neg_class] = -1
    y_test[y_test == pos_class] = 1
    y_test[y_test == neg_class] = -1

    logger.debug("x_train shape: %s", x_train.shape)
    logger.debug("y_train shape: %s", y_train.shape)

    return (x_train, y_train), (x_test, y_test)

def load_data(dataset: str, logger: Logger, **kwargs):
    """Loads the given dataset.
    
    Parameters
    ----------
    dataset : str
        Name of the dataset to load.
    logger : Logger
        Logger to use.
    
    Returns
    -------
    np.ndarray
        Loaded dataset.
    """
    import tensorflow as tf
    
    DATASETS = {
        "mnist": tf.keras.datasets.mnist,
        "cifar10": tf.keras.datasets.cifar10,
        "fashionmnist": tf.keras.datasets.fashion_mnist,
    }
    if dataset in DATASETS:
        (x_train, y_train), (x_test, y_test) = DATASETS[dataset].load_data()
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))
    return preproc_data(x_train, y_train, x_test, y_test, kwargs['subset'], kwargs['pos_class'], kwargs['neg_class'], logger)

def make_dataset(url:str, download:bool=True):
    
    if "data" not in os.listdir():
        os.makedirs("data")
    
    hval = str(hashlib.sha256(url.encode()).hexdigest())
    
    if "download_log" in os.listdir():
        with open("download_log", "r") as file:
            
            if str(hval) in file.readlines():
                print(f"Dataset already downloaded: download_log -> {hval}")
                return False
    else:
        with open("download_log", "w") as file:
            file.write(hval)
    
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser') 
    else:
        print("Failed to retrieve data from website")
    
    res = [i.text.strip().replace("\n","").replace("\r","") for i in soup.find_all('p')]
    
    fname =  res[1] + ".txt"
    
    res = "\n".join(res)
    
    with open(os.path.join("data/",fname), "w") as file:
        file.write(res)
    
    print("Dataset successfully downloaded")
    return True

def build_vocab():
    #lowercase
    alphabet_dict = {chr(i + 97): i + 1 for i in range(26)}
    alphabet_dict.update({chr(i + 65): i + 27 for i in range(26)})

    special_characters = [".", ",", "!", "?", "'", ":", ";", "-", "_"]
    special_characters_dict = {char: i + 53 for i, char in enumerate(special_characters)}
    alphabet_dict.update(special_characters_dict)

    alphabet_dict[" "] = 0
    return alphabet_dict

def tokenizer(fp:str):
    vocab = build_vocab()
    
    with open(fp) as file:
        data = file.readlines()
    res = np.array([])

    for i in data:
        if not i:
            continue
        
        tokens = np.array([0]*128)
        for j in range(len(tokens)):
            if j >= len(i):
                break
            if i[j] in vocab:
                tokens[j] = vocab[i[j]]
        
        
        res = np.append(res,tokens)

    res.reshape(len(data),128)
    
    return res