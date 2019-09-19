import numpy as np
import pandas as pd

from itertools import product
from explore.ContCont import ContCont


def data_iter():

    n = 20
    a = np.random.normal(size=n)
    b = np.random.normal(size=n)

    yield a, b

    a = pd.Series(a, index=np.arange(n).astype(str))
    b = pd.Series(b, index=np.arange(n).astype(str))
    yield a, b

    a.name = 'a'
    b.name = 'b'
    yield a, b

    a.iloc[0] = np.nan
    yield a, b


def settings_iter():
    settings = {'alpha': 0.05}
    for measure in ['pearson', 'spearman']:
        settings['measure'] = measure
        yield settings


def test_settings():
    """
    Makes sure different settings of ContCont run.
    """

    for (a, b), settings in product(data_iter(),
                                    settings_iter()):
        test = ContCont(**settings)
        test = test.fit(a, b)
        test.plot()
        assert True

    test.plow_kws = {'standardize': True}
    test.plot()
    assert True
