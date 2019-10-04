import pandas as pd
import numpy as np
from itertools import product, combinations
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from textwrap import dedent
import seaborn as sns

from explore.utils import is_cat
from explore.PairwiseTests import PairwiseTestsMixin, _pairwise_docs
from explore.viz.utils import count_plot
from explore.viz.jitter import jitter


class SingleBlock(PairwiseTestsMixin):

    def fit(self, X):
        """
        Parameters
        ----------
        X: array-like, (n_samples, n_features)
            The data block whose variables we wish to compare.
        """
        self.X_ = pd.DataFrame(X)

        self.var_names_ = np.array(self.X_.columns)

        # TODO: it would be nice if comparisons_ were a pd.Series with
        # set indices, however this casues weired errors from .loc[].
        # try to figure this out.
        self.comparisons_ = {}
        for row_var, col_var in combinations(self.var_names_, 2):
            r = self.X_[row_var]
            c = self.X_[col_var]

            tst = self._get_test(r, c)

            self.comparisons_[frozenset((row_var, col_var))] = tst

        return self

    def test_iter(self):
        for a_var, b_var in combinations(self.var_names_, 2):
            yield self.comparisons_[frozenset((a_var, b_var))]

    def plot(self, inches=8, verbosity=1):
        """
        Makes an upper trianglular grid of plots. The i,j th off diagaonal
        plot compares variable i to variable j. The diagonal plots
        summarize each variable.


        Parameters
        ----------
        inches: float, None
            Argument for plt.figsize, if None, will not create a new plt.figure.

        verbosity: int
            Amount of detail to include in the plot.
        """

        n_rows = len(self.var_names_)
        n_cols = n_rows

        if inches is not None:
            plt.figure(figsize=(inches * n_cols, inches * n_rows))

        grid = gridspec.GridSpec(n_rows, n_cols)
        for i, j in product(range(n_rows), range(n_cols)):
            row_var = self.var_names_[i]
            col_var = self.var_names_[j]

            if i < j:
                tst = self.comparisons_[frozenset((row_var, col_var))]

                plt.subplot(grid[i, j])
                tst.plot(verbosity=verbosity)

            elif i == j:
                values = self.X_[row_var]

                plt.subplot(grid[i, j])
                if is_cat(values):
                    count_plot(values)
                else:
                    values = values.dropna()  # remove missing values for sns.distplot

                    sns.distplot(values, color='black',
                                 kde_kws={'alpha': .2, 'shade': True,
                                          'zorder': 0},
                                 hist_kws={'alpha': .2, 'zorder': 1})

                    jitter(values, zorder=2)
                    plt.xlabel(row_var)
                    plt.ylim(0)


SingleBlock.__doc__ = dedent("""\
    Compares all pairs of variables in a single data block.

    Parameters
    ----------
    {pairwise_params}

    Attributes
    ----------
    comparisons_: dict of Test/TestCollections
        All pairwise comparisons between variables in this datablock.
        Note the keys of comparisons_ are frozenset((var_a, var_b)).

    var_names_: array-like
        The names of each variable.

    """).format(**_pairwise_docs)
