import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency

from explore.Base import TestMixin
from explore.utils import get_crosstab, get_counts, process_var_pair, is_cat
from explore.viz.utils import bold, fmt_pval
from explore.viz.cross_categories import plot_cross_counts, plot_cross_props, to_cat_df
from explore.viz.utils import heatmap


class CatCat(TestMixin):
    """
    Categorical vs. categorical comparison.

    Parameters
    ----------
    alpha: float
        Significance level.

    vis_mode: str
        Must be one of ['count', 'resit', 'chi2']

    nan_how: str
        How to handle nan values in the categorical variable. Must be one
        of ['drop', 'leave']. If 'leave', nan values are treated as a category.

    """

    def __init__(self, alpha=0.05, nan_how='drop'):
        self.alpha = alpha
        self.nan_how = nan_how

        self.plot_mode = 'count'

    def fit(self, a, b):
        """
        Parameters
        ----------
        a, b: array-like, (n_samples, )
            Categorical variable.
        """
        if not is_cat(a):
            raise ValueError('a must be a categorical variable.')
        if not is_cat(b):
            raise ValueError('b must be a categorical variable.')

        self.a_, self.a_nan_idxs_, self.b_, self.b_nan_idxs_ = \
            process_var_pair(a, b,
                             nan_how_a=self.nan_how,
                             nan_how_b=self.nan_how)
        self.var_names_ = [self.a_.name, self.b_.name]

        self.a_ = self.a_.astype(str)
        self.b_ = self.b_.astype(str)
        self.labels_a_ = np.unique(self.a_)
        self.labels_b_ = np.unique(self.b_)

        self.counts_a_ = get_counts(self.a_)
        self.counts_b_ = get_counts(self.b_)

        self.n_cats_a_ = len(self.labels_a_)
        self.n_cats_b_ = len(self.labels_b_)

        self.cross_ = get_crosstab({self.var_names_[0]: self.a_,
                                    self.var_names_[1]: self.b_})
        # self.table_ = sm.stats.Table(cross)
        chi2, pval, dof, expected = chi2_contingency(self.cross_)
        self.chi2_ = chi2
        self.pval_raw_ = pval
        self.dof_ = dof

        return self

    @property
    def clust_df_(self):
        return to_cat_df({self.var_names_[0]: self.a_,
                          self.var_names_[1]: self.b_})

    def plot(self, verbosity=1):
        """
        Plots a contingency table.

        Parameters
        ----------
        verbosity: int
            Amount of detail to include in the plot.
        """

        if self.plot_mode == 'count':
            values = self.cross_
            fmt = 'd'
            center = None
        elif self.plot_mode == 'resid':
            table = sm.stats.Table(self.cross_)
            values = table.resid_pearson
            fmt = '1.3f'
            center = 0
        elif self.plot_mode == 'chi2':
            table = sm.stats.Table(self.cross_)
            values = table.chi2_contribs
            fmt = '1.3f'
            center = None
        else:
            raise ValueError("plot_mode must be one of ['count', 'resid', 'chi2']")

        # sns.heatmap(values, annot=True, fmt=fmt, center=center)
        heatmap(values, annot=True, fmt=fmt, center=center)
        # TODO: switch to sns.heatmap when https://github.com/mwaskom/seaborn/issues/1773 gets fixed

        title = '(chi2={:1.3f}, p={})'.format(self.chi2_,
                                              fmt_pval(self.pval_))
        if self.rejected_:
            title = bold('dependent* ') + title
        else:
            title = 'independent ' + title

        plt.title(title)

    def plot_cross_counts(self, val='count', fig_kws={}):
        plot_cross_counts(self.clust_df_, val=val, fig_kws=fig_kws)

    def plot_cross_props(self, fig_kws={}):
        plot_cross_props(self.clust_df_, fig_kws=fig_kws)
