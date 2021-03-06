import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from textwrap import dedent

from explore.PairwiseTests import PairwiseTestsMixin, _pairwise_docs


class BlockBlock(PairwiseTestsMixin):

    def fit(self, R, C):
        """
        Parameters
        ----------
        R: array-like, (n_samples, n_row_vars)
            The data block whose variables go on the rows.

        C: array-like, (n_samples, n_col_vars)
            The data block whose variables go on the columns.
        """
        R = pd.DataFrame(R)
        C = pd.DataFrame(C)
        assert all(R.index == R.index)

        self.row_var_names_ = np.array(R.columns)
        self.col_var_names_ = np.array(C.columns)

        self.comparisons_ = pd.DataFrame(index=self.row_var_names_,
                                         columns=self.col_var_names_)
        for row_var, col_var in product(self.row_var_names_,
                                        self.col_var_names_):
            r = R[row_var]
            c = C[col_var]

            tst = self._get_test(r, c)

            self.comparisons_.loc[row_var, col_var] = tst

        return self

    def test_iter(self):
        for row_var, col_var in product(self.row_var_names_,
                                        self.col_var_names_):
            yield self.comparisons_.loc[row_var, col_var]

    def plot(self, inches=8, verbosity=1, wspace=None, hspace=None):
        """
        Makes a grid of plots where the i, j th plot shows the comparison between the ith variable of the first dataset and the jth variable of the second dataset.
        """

        nrows = len(self.row_var_names_)
        ncols = len(self.col_var_names_)

        if inches is not None:
            plt.figure(figsize=(inches * ncols, inches * nrows))

        grid = GridSpec(nrows=nrows, ncols=ncols,
                        wspace=wspace, hspace=hspace)

        for i, j in product(range(nrows), range(ncols)):
            row_var = self.row_var_names_[i]
            col_var = self.col_var_names_[j]

            tst = self.comparisons_.loc[row_var, col_var]

            plt.subplot(grid[i, j])
            tst.plot(verbosity=verbosity)


BlockBlock.__doc__ = dedent("""\
    Compares all pairs of variables between two data blocks.

    Parameters
    ----------
    {pairwise_params}

    Attributes
    ----------
    comparisons_: pd.DataFrame of Test/TestCollections
        Tests comparing variable parirs between each dataset.

    row_var_names_, col_var_names_: np.array
        The variable names of the first/second data blocks.

    """).format(**_pairwise_docs)
