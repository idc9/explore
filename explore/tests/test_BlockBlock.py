import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from itertools import product
import matplotlib.pyplot as plt

from explore.BlockBlock import BlockBlock
from explore.tests.test_SingleBlock import settings_iter


def data_iter():

    np.random.seed(2342)

    n = 20
    n_cts = 3

    A = pd.DataFrame(np.random.normal(size=(n, n_cts)))
    A['cat'] = np.random.choice([0, 1], size=n).astype(str)

    B = pd.DataFrame(np.random.normal(size=(n, n_cts)))
    B['dog'] = np.random.choice([0, 1, 2], size=n).astype(str)

    for A, B in format_iter(A, B):
        yield A, B


def format_iter(A, B):
    """
    Iterates over various formats
    """

    yield A, B

    A.iloc[0, 1] = np.nan
    yield A


def test_settings():

    for (A, B), settings in product(data_iter(),
                                    settings_iter()):

        tests = BlockBlock(**settings)
        tests = tests.fit(A, B)
        tests.plot()
        plt.close()
        assert True
