import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from itertools import product

from explore.ContCat import ContCat


def data_iter():

    np.random.seed(2342)

    n = 30

    # 2 classes
    cont = np.random.normal(size=n)
    cat = np.random.choice([0, 1], size=n).astype(str)

    for cont, cat in format_iter(cont, cat):
        yield cont, cat

    # 3 classes
    cont = np.random.normal(size=n)
    cat = np.random.choice([0, 1, 3], size=n).astype(str)

    for cont, cat in format_iter(cont, cat):
        yield cont, cat


def format_iter(cont, cat):
    """
    Iterates over various formats
    """

    yield cont, cat

    cont = pd.Series(cont)
    cat = pd.Series(cat)
    yield cont, cat

    cat = cat.astype('category')
    yield cont, cat

    cont.name = 'continuous'
    cat.name = 'mr mittens'
    yield cont, cat

    cont.iloc[0] = np.nan
    cat.iloc[1] = np.nan
    yield cont, cat


def settings_iter():
    settings = {'alpha': [0.05],
                'test': ['auc', 't'],
                'multi_cat': ['ovo', 'ovr'],
                'nan_how': ['drop', 'leave']}

    for setting in ParameterGrid(settings):
        yield setting


def test_settings():
    """
    Makes sure different settings of ContCont run.
    """

    for (cont, cat), settings in product(data_iter(),
                                         settings_iter()):

        test = ContCat(**settings)
        test = test.fit(cont, cat)
        test.plot()
        assert True
