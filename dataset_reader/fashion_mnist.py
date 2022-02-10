import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from scipy.special import erf

"""
x_train = dataset['train']['data']
    y_train = dataset['train']['label']
    x_test = dataset['validation']['data']
    y_test = dataset['validation']['label']
 """


def load_dataset(flatten=True):
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    """

    # Iwdfs preprocessing
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3] if len(x_train.shape) > 3 else 1))
    #x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3] if len(x_test.shape) > 3 else 1))
    if flatten:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    x_train = (x_train.astype('float32') / 255. - 0.5) * 2
    x_test = (x_test.astype('float32') / 255. - 0.5) * 2

    dataset = {}
    dataset['train'] = {}
    dataset['validation'] = {}
    dataset['train']['data'] = x_train
    dataset['train']['label'] = y_train
    dataset['validation']['data'] = x_test
    dataset['validation']['label'] = y_test
    return dataset


class Normalize:  # we are mocking up this for mnist for now TODO review/replace/or remove completely

    def __init__(self):
        self.stats = None

    def fit(self, X):
        """
        mean = np.mean(X, axis=0)
        std = np.sqrt(np.square(X - mean).sum(axis=0) / max(1, len(X) - 1))
        self.stats = (mean, std)
        """

    def transform(self, X):
        #transformed_X = erf((X - self.stats[0]) / (np.maximum(1e-6, self.stats[1]) * np.sqrt(2.)))
        transformed_X = X
        return transformed_X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
