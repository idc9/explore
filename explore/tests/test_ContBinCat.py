import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from itertools import product

from explore.ContBinCat import ContBinCat


def data_iter():

    np.random.seed(2342)

    n = 20
    cont = np.random.normal(size=n)
    cat = np.random.choice([0, 1], size=n).astype(str)

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
                'test': ['ad', 'auc', 'mw', 'ks', 't']}

    for setting in ParameterGrid(settings):
        yield setting


def test_settings():
    """
    Makes sure different settings of ContCont run.
    """

    for (cont, cat), settings in product(data_iter(),
                                         settings_iter()):

        test = ContBinCat(**settings)
        test = test.fit(cont, cat)
        test.plot()
        assert True
