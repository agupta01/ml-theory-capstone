import logging
from typing import List

import numpy as np
#from sklearn.datasets import make_classification
import hashlib
from src import utils
import os
import requests
import json
from bs4 import BeautifulSoup

import PyPDF2

TARGET_FNS = {"xsinx": utils.target_xsinx}


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] \t %(message)s",
    datefmt="%b %d %Y %I:%M%p",
)


def generate_test_data(
    target_fn_id: str, n_samples: int, noise_std: float, save_path: str = None
):
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


def generate_test_classification(
    n_samples: int,
    n_features: int,
    n_classes: int,
    informative_pct: float = 1.0,
    save_path: str = None,
):
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
    informative_features = int(informative_pct * n_features)
    noise_features = max(n_features - informative_features, 0)
    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=informative_features,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        random_state=0,
    )
    data = np.hstack([x, y.reshape(-1, 1)])
    if save_path:
        np.savetxt(save_path, data, delimiter=",")
    return data


def preproc_data(
    x_train,
    y_train,
    x_test,
    y_test,
    subset: float,
    pos_class: int,
    neg_class: int,
    logger: logging.Logger,
):
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


def load_data(dataset: str, logger: logging.Logger, **kwargs):
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
    return preproc_data(
        x_train,
        y_train,
        x_test,
        y_test,
        kwargs["subset"],
        kwargs["pos_class"],
        kwargs["neg_class"],
        logger,
    )


def make_dataset(url: str, fname: str, download: bool = True):
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
        soup = BeautifulSoup(response.text, "html.parser")
    else:
        print("Failed to retrieve data from website")

    res = [
        i.text.strip().replace("\n", "").replace("\r", "") for i in soup.find_all("p")
    ]

    fname = fname + ".txt"
    
    res = "\n".join(res)
    
    with open(os.path.join("data/", fname), "w") as file:
        file.write(res)

    print("Dataset successfully downloaded")
    return True

def make_all_datasets(urls: List[str]):
    for i in range(len(urls)):
        make_dataset(urls[i],str(i))
    return True
    
def build_vocab():
    # lowercase
    alphabet_dict = {chr(i + 97): i + 1 for i in range(26)}
    # alphabet_dict.update({chr(i + 65): i + 27 for i in range(26)})
    numbers = [str(i) for i in range(10)]
    numbers_dict = {num: i + 27 for i, num in enumerate(numbers)}
    alphabet_dict.update(numbers_dict)

    special_characters = [".", ",", "!", "?", "’", ":", ";", "-", "_", "&", "(", ")"]
    special_characters_dict = {
        char: i + 37 for i, char in enumerate(special_characters)
    }
    alphabet_dict.update(special_characters_dict)

    alphabet_dict[" "] = 0
    alphabet_dict["<UNK>"] = 49

    return alphabet_dict


def tokenizer(fp: str, contextsize: int = 32):
    vocab = build_vocab()
    unknown_chars = set()

    with open(fp) as file:
        data = file.readlines()

    res = np.array([])
    line_ct = len(data)
    for i in data:
        if not i:
            continue
        
        i = i.lower()

        context = np.array([np.array([0] * len(vocab))] * contextsize)

        if len(i) < contextsize:
            line_ct -= 1
            continue
        
        for j in range(contextsize):

            if i[j] in vocab:
                context[j][int(vocab[i[j]])] = 1

            else:
                if ord(i[j]) > 127 or ord(i[j]) == 10:
                    pass
                else:
                    context[j][int(vocab["<UNK>"])] = 1
                    # log the ascii code of the unknown character
                    unknown_chars.add(ord(i[j]))

        res = np.append(res, context)

    res = res.reshape(line_ct, contextsize, len(vocab))

    logger.info(f"Unknown characters: {unknown_chars}")

    return res


def pdf_tokenizer(fp: str, contextsize: int = 32):
    
    vocab = build_vocab()
    
    unknown_chars = set()

    pdf_file = open(fp,"rb")
    
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Extract text from each page in the PDF file
    corpus = ""
    
    for page in range(len(pdf_reader.pages)):
        page_obj = pdf_reader.pages[page]
        text = page_obj.extract_text()
        corpus += text.replace("\n"," ")
    # Close the PDF file
    pdf_file.close()
    
    
    lines = []
    for i in range(0,len(corpus),contextsize):
        if i + contextsize >= len(corpus):
            break
        lines.append(corpus[i:i+contextsize])
        
    
    res = np.array([])
    
    line_ct = 0
    
    for i in lines:
        
        if not i:
            continue
        
        if len(i) < contextsize:
            continue
        
        i = i.lower()

        context = np.array([np.array([0] * len(vocab))] * contextsize)
        line_ct += 1
        
        for j in range(contextsize):

            if i[j] in vocab:
                context[j][int(vocab[i[j]])] = 1

            else:
                if ord(i[j]) > 127 or ord(i[j]) == 10:
                    pass
                else:
                    context[j][int(vocab["<UNK>"])] = 1
                    # log the ascii code of the unknown character
                    unknown_chars.add(ord(i[j]))

        res = np.append(res, context)

    res = res.reshape(line_ct, contextsize, len(vocab))

    logger.info(f"Unknown characters: {unknown_chars}")

    return res


def generate_corpus(contextsize:int=64):
    fps = os.listdir("data")

    for i in range(len(fps)):
        if i == 0:
            corpus = tokenizer(os.path.join("data",fps[i]),contextsize)
            continue
        
        corpus = np.concatenate((corpus,tokenizer(os.path.join("data",fps[i]),contextsize)), axis = 0)
        
    return corpus