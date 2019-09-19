import pandas as pd
import numpy as np
from collections import Counter
from joblib import Parallel, delayed


def process_var_pair(a, b, nan_how_a='drop', nan_how_b='drop'):
    """
    Processes a pair of variables
    - convert to named pandas series
    - handle nans
    - check both have same indices
    """
    assert nan_how_a in ['drop', 'leave']
    assert nan_how_b in ['drop', 'leave']

    var_names = [0, 1]
    if hasattr(a, 'name') and a.name is not None:
        var_names[0] = a.name
    if hasattr(b, 'name') and b.name is not None:
        var_names[1] = b.name

    # make sure pd.Series
    a = pd.Series(a, name=var_names[0])
    b = pd.Series(b, name=var_names[1])

    # make sure observations are aligned
    assert all(a.index == b.index)

    a_nan_idxs = a.index[a.isna()]
    b_nan_idxs = b.index[b.isna()]

    # drop observations where either variable is nan
    to_drop = []
    if nan_how_a == 'drop':
        to_drop += list(a_nan_idxs)
    if nan_how_b == 'drop':
        to_drop += list(b_nan_idxs)

    a = a.drop(to_drop)
    b = b.drop(to_drop)

    return a, a_nan_idxs, b, b_nan_idxs


def devec(v):
    """
    Turns a pd.Series which is indexed by tuples into a pd.DataFrame
    """
    rows, cols = zip(*list(v.index))
    labels = safe_unique(np.concatenate([rows, cols]))

    mat = pd.DataFrame(index=labels, columns=labels)

    for (a, b), value in v.iteritems():
        mat.loc[a, b] = value
        mat.loc[b, a] = value

    return mat

# TODO: delete
# def dod_mat(row_keys, col_keys):
#     """
#     Returns an empty matrix of dictsionaries.
#     """
#     return {rk: {ck: None for ck in col_keys} for rk in row_keys}


def is_cat(x):
    """
    Returns true of x is categorical.
    """
    # TODO: more elegant/better way to do this
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        if isinstance(x.dtype, pd.CategoricalDtype):
            return True

    x = np.array(x)
    # elif not isinstance(x, np.array):
    #     x = np.array(x)

    return not np.issubdtype(x.dtype, np.number)


def get_counts(y):
    """
    Counts the number of smaples in each class.

    Parameters
    ----------
    y: array-like, (n_samples, )
        Class labels for each observation.

    Output
    ------
    y_cnts: pd.Series
    """
    return pd.Series(Counter(y)).sort_values(ascending=False)


def get_crosstab(d):
    """
    Gets cross tab matrix.

    Parameters
    ----------
    dict of lists
    """
    keys = list(d.keys())
    return pd.crosstab(pd.Categorical(d[keys[0]]),
                       pd.Categorical(d[keys[1]]),
                       rownames=[keys[0]],
                       colnames=[keys[1]])


def safe_unique(ar):
    """
    Basically np.unique, but will not duplcate np.nans

    # unsafe unique
    np.unique([np.nan, np.nan])
    >>> np.array([np.nan, np.nan])

    # safe unique
    safe_unique([np.nan, np.nan])
    >>> np.array([np.nan])
    """
    # return np.array(list(set(ar)))
    # return np.array(sorted(set(np.array(ar))))
    labels = np.unique(ar)
    nan_mask = np.isnan(labels)
    if sum(nan_mask) > 0:
        labels = labels[~nan_mask]
        labels = np.append(labels, np.nan)

    return labels


def has_and_not_none(obj, name):
    """
    Returns True iff obj has attribute name and obj.name is not None
    """
    return hasattr(obj, name) and (getattr(obj, name) is not None)


def safe_apply(f, x):
    """
    Applies a function f to x. If x was a pandas object, will return
    the same pandas object.
    Parameters
    ----------
    f: callable
        The function.

    x: pd.DataFrame, pd.Series
    """
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(f(x), index=x.index, columns=x.columns)
    elif isinstance(x, pd.Series):  # TODO: test this
        return pd.Series(f(x), index=x.index, name=x.name)
    else:
        return f(x)


def Proportions(y):
    """
    Returns Counter(y) as proportions
    """
    c = Counter(y)
    n = len(y)
    return {k: c[k] / n for k in c.keys()}


def sample_parallel(fun, n_samples, n_jobs=None, *args, **kwargs):
    """
    Computes samples possibly in parralel using from sklearn.externals.joblib


    Parameters
    ---------
    fun: callable
        The sampling function.

    n_samples: int
        Number of samples to draw.

    n_jobs: None, -1, int
        Number of cores to use. If None, will not sample in parralel.
         If -1 will use all available cores.

    *args: args for fun

     **kwargs:  key word args for fun

    Output
    ------
    samples: list
        Each entry of samples is the output of one call to
        fun(*args, **kwargs)
    """

    if n_jobs is not None:
        return Parallel(n_jobs=n_jobs)(delayed(fun)(*args, **kwargs)
                                       for s in range(n_samples))

    else:
        return [fun(**kwargs) for s in range(n_samples)]
