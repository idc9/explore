import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from itertools import product
import matplotlib.pyplot as plt

from explore.SingleBlock import SingleBlock


def data_iter(seed=2342):

    np.random.seed(seed)

    n = 20
    n_cts = 3

    X = np.random.normal(size=(n, n_cts))
    for X in format_iter(X):
        yield X

    a = np.random.choice([0, 1], size=n).astype(str)
    b = np.random.choice([0, 1, 3], size=n).astype(str)
    X = pd.DataFrame(X)
    X['a'] = a
    X['b'] = b

    for X in format_iter(X):
        yield X


def format_iter(X):
    """
    Iterates over various formats
    """

    yield X

    X = pd.DataFrame(X)
    yield X

    X.iloc[0, 1] = np.nan
    yield X


def settings_iter():
    settings = {'nan_how': ['drop', 'leave'],
                'multi_test': ['fdr_bh', 'bonferroni', 'fdr_tsbh']}

    for setting in ParameterGrid(settings):
        yield setting


def test_settings():
    """
    Makes sure different settings of ContCont run.
    """

    for X, settings in product(data_iter(),
                               settings_iter()):

        tests = SingleBlock(**settings)
        tests = tests.fit(X)
        tests.plot()
        plt.close()
        assert True

