import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from itertools import product

from explore.CatCat import CatCat


def data_iter():

    np.random.seed(2342)

    n = 30

    # 2 classes
    a = np.random.choice([0, 1], size=n).astype(str)
    b = np.random.choice([0, 1, 3], size=n).astype(str)

    for a, b in format_iter(a, b):
        yield a, b


def format_iter(a, b):
    """
    Iterates over various formats
    """

    yield a, b

    a = pd.Series(a)
    b = pd.Series(b)
    yield a, b

    a = a.astype('category')
    yield a, b

    a.name = 'woof'
    b.name = 'meow'
    yield a, b

    a.iloc[0] = np.nan
    b.iloc[1] = np.nan
    yield a, b


def settings_iter():
    settings = {'alpha': [0.05],
                'nan_how': ['drop', 'leave']}

    for setting in ParameterGrid(settings):
        yield setting


def test_settings():
    """
    Makes sure different settings of ContCont run.
    """

    for (cont, cat), settings in product(data_iter(),
                                         settings_iter()):

        test = CatCat(**settings)
        test = test.fit(cont, cat)
        test.plot()
        assert True
