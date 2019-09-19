"""
Visualization functions to compare different categorical variables for a
fixed set of observations. E.g. suppose we have n = 100 patients and
for each patient we have disease type, home state, and gender.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from itertools import product

from hdda.viz.utils import count_plot


def plot_cross_props(categories, fig_kws={}):
    """
    Visualizes proportion of observations in category i which are also
    in category j

    Parameters
    ----------
    categories: pd.DataFrame, (n_samples, n_categories)
        A data frame where rows correspond to samples and columns
        are different kinds of categories.

    fig_kws: dict
        Key word arguments for plt.figure.
    """

    categories = pd.DataFrame(categories)
    cats = categories.columns
    n_cats = len(cats)

    plt.figure(figsize=[n_cats * 5, n_cats * 5], **fig_kws)

    for plt_idx, (i, j) in enumerate(product(range(n_cats), range(n_cats))):
        if i != j:

            cross = pd.crosstab(categories[cats[i]], categories[cats[j]],
                                normalize='index')  # * 100

            plt.subplot(n_cats, n_cats, plt_idx + 1)
            sns.heatmap(cross, annot=True, fmt='1.3f')

        else:
            plt.subplot(n_cats, n_cats, plt_idx + 1)
            count_plot(categories[cats[i]], display='prop')


def plot_cross_counts(categories, val='count', fig_kws={}):
    """
    Makes a grid of plots to compare relations between different
    categorical variables. Each row/column corresoponds to a categorical
    variable i.e. row i, column j compares variables i and j.
    The ith diagonal plots shows the class counts for the ith category.
    The i, jth off digaonal plot shows a heatmap comparing the ith to the
    jth category. The values on of the heatmap show a user selected metric
    for how similar the classes are.

    Parameters
    ----------
    categories: pd.DataFrame, (n_samples, n_categories)
        A data frame where rows correspond to samples and columns
        are different kinds of categories.

    fig_kws: dict
        Key word arguments for plt.figure.

    val: str, ['count', 'resid', 'chi2']
        Heatmaps will show either raw counts, pearson residuals or chi2 statistics
    """
    # TODO: add pairwise adjusted random scores
    assert val in ['count', 'resid', 'chi2']

    categories = pd.DataFrame(categories)

    cats = categories.columns
    n_cats = len(cats)

    fmt = 'd'
    center = None

    plt.figure(figsize=[n_cats * 6, n_cats * 5], **fig_kws)
    for plt_idx, (i, j) in enumerate(product(range(n_cats), range(n_cats))):

        # off digonal terms compare two different categorical variables.
        if i < j:
            cross = pd.crosstab(categories[cats[i]], categories[cats[j]])

            if val == 'resid':
                table = sm.stats.Table(cross)
                cross = table.resid_pearson
                fmt = '1.3f'
                center = 0

            elif val == 'chi2':
                table = sm.stats.Table(cross)
                cross = table.chi2_contribs
                fmt = '1.3f'

            elif val == 'pairwise_ars':
                raise NotImplementedError

            plt.subplot(n_cats, n_cats, plt_idx + 1)
            sns.heatmap(cross, annot=True, fmt=fmt, center=center)

        # plot category counts on diagonal
        elif i == j:
            plt.subplot(n_cats, n_cats, plt_idx + 1)
            # sns.countplot(categories[cats[i]], orient='v')
            count_plot(categories[cats[i]])

        # put title on first off diag plot
        if i == 0 and j == 1:
            title = {'resid': 'pearson residuals',
                     'chi2': 'chi sq contributions',
                     'count': 'count',
                     'pairwise_ars': 'pairwise adjusted rand score'}
            plt.title(title[val])

        if i == 0 and j == 0:
            plt.title('class counts')


def to_cat_df(categories):
    """
    Takes a list or dict of categorical variables and returns a pandas dataframe.
    """
    if isinstance(categories, dict):
        return pd.DataFrame.from_dict(categories, dtype='category')
    else:
        return pd.DataFrame(categories, dtype='category').T
