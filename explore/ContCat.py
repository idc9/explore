import pandas as pd
import numpy as np
from copy import deepcopy
from itertools import combinations

from explore.Base import TestCollectionMixin
from explore.ContBinCat import ContBinCat
from explore.utils import process_var_pair, is_cat, get_counts
from explore.ubdt import _get_ovr_label, _get_ovo_labels
from explore.viz.cat_distplot import cat_distplot, cat_boxplot


class ContCat(TestCollectionMixin):
    """
    Continuous vs. categorical variable comparision.

    Parameters
    ----------
    alpha: float
        Significance level.

    test: str
        Which difference in distribution test to use, see
        swdb.compare.univariate_binary_dists_tests.binary_dist_test.

    multi_cat: str
        Which pairwise comparisons to make for multiple categories (one-vs-one or one-vs-rest). Must be one of ['ovo', 'ovr'].

    nan_how: str
        How to handle nan values in the categorical variable. Must be one
        of ['drop', 'leave']. If 'leave', nan values are treated as a category.
        Note continuous nan values are always dropped.

    Attributes
    ----------
    comparisons_
    """

    def __init__(self, alpha=0.05, test='auc', multi_cat='ovo',
                 nan_how='drop'):
        self.alpha = alpha
        self.test = test
        self.multi_cat = multi_cat
        self.nan_how = nan_how

        self.plot_kws = {}  # keyword args to cat_distplot
        self.plot_mode = 'distplot'

    def fit(self, cont, cat):
        """
        Parameters
        ----------
        cont: array-like, (n_samples, )
            Continuous variable.

        cat: array-like, (n_samples, )
            Categorical variable.
        """
        if is_cat(cont):
            raise ValueError('a must be a continuous variable.')
        if not is_cat(cat):
            raise ValueError('b must be a categorical variable.')

        self.cont_, self.cont_nan_idxs_, self.cat_, self.cat_nan_idxs_ = \
            process_var_pair(a=cont, b=cat,
                             nan_how_a='drop',
                             nan_how_b=self.nan_how)
        self.var_names_ = [self.cont_.name, self.cat_.name]

        self.cat_ = self.cat_.astype(str)
        self.labels_ = np.unique(self.cat_)
        self.counts_ = get_counts(self.cat_)
        self.n_cats_ = len(self.labels_)

        if self.n_cats_ == 2:
            self.comparisons_ = ContBinCat(alpha=self.alpha, test=self.test)
            self.comparisons_.fit(self.cont_, self.cat_)

        elif self.multi_cat == 'ovo':
            self.comparisons_ = {}
            for cl_0, cl_1 in combinations(self.labels_, 2):
                values_0 = self.cont_[self.cat_ == cl_0]
                values_1 = self.cont_[self.cat_ == cl_1]
                # values_0 = self.cont_[nansafe_equal_mask(self.cat_, cl_0)]
                # values_1 = self.cont_[nansafe_equal_mask(self.cat_, cl_1)]
                values = np.concatenate([values_0, values_1])
                classes = np.concatenate([[cl_0] * self.counts_[cl_0],
                                          [cl_1] * self.counts_[cl_1]])

                classes = pd.Series(classes,
                                    dtype='category',
                                    name=self.cat_.name)

                tst = ContBinCat(alpha=self.alpha, test=self.test)
                tst.fit(values, classes)

                self.comparisons_[frozenset((cl_0, cl_1))] = tst

        elif self.multi_cat == 'ovr':
            self.comparisons_ = {}
            for cl in self.labels_:
                b_ovr = deepcopy(self.cat_)
                # b_ovr[~nansafe_equal_mask(self.cat_, cl)] = 'not_{}'.format(cl)
                b_ovr[self.cat_ != cl] = 'not_{}'.format(cl)
                tst = ContBinCat(alpha=self.alpha, test=self.test)
                tst.fit(self.cont_, b_ovr)

                self.comparisons_[cl] = tst

        else:
            raise ValueError('multi_cat must be one of ["ovo", "ovr"]')

        return self

    def test_iter(self):
        if self.n_cats_ == 2:
            yield self.comparisons_

        elif self.multi_cat == 'ovr':
            for cl in self.labels_:
                yield self.comparisons_[cl]

        elif self.multi_cat == 'ovo':
            for cl_0, cl_1 in combinations(self.labels_, 2):
                yield self.comparisons_[frozenset((cl_0, cl_1))]

    def get_cl_ovo_results(self, cl):
        """
        Returns a pd.DataFrame sumarizing the comparision results for
        one class.

        Parameters
        ----------
        cl:
            The class label.
        """
        if self.multi_cat != 'ovo':
            raise ValueError()

        results = pd.DataFrame(columns=['stat', 'rejected', 'pval'])
        for other_cl in self.labels_:
            if other_cl != cl:
                tst = self.comparisons_[frozenset((cl, other_cl))]
                results.loc[other_cl, 'stat'] = tst.stat_
                results.loc[other_cl, 'pval'] = tst.pval_
                results.loc[other_cl, 'rejected'] = tst.rejected_
        return results

    def get_ovr_results(self):
        """
        Returns a pd.DataFrame sumarizing the comparision results for all one-vs-rest comparisons.

        """
        if self.multi_cat != 'ovr':
            raise ValueError()

        results = pd.DataFrame(index=self.labels_,
                               columns=['stat', 'rejected', 'pval'])
        for cl in self.labels_:
            tst = self.comparisons_[cl]
            results.loc[cl, 'stat'] = tst.stat_
            results.loc[cl, 'pval'] = tst.pval_
            results.loc[cl, 'rejected'] = tst.rejected_

        return results

    def plot(self):
        """
        Plots a categorical histogram.
        """

        cl_labels = {}
        if self.n_cats_ == 2:
            self.comparisons_.plot()

        else:
            for cl in self.labels_:
                if self.multi_cat == 'ovr':
                    bin_test = self.comparisons_[cl]
                    label_kws = {'cl': cl,
                                 'n': self.counts_[cl],
                                 'stat_name': self.test,
                                 'stat': bin_test.stat_,
                                 'reject': bin_test.rejected_,
                                 'test_prefix': 'one-vs-rest'}

                    cl_labels[cl] = _get_ovr_label(**label_kws)

                elif self.multi_cat == 'ovo':
                    cl_labels = _get_ovo_labels(self)

            if self.plot_mode == 'distplot':
                cat_distplot(values=self.cont_,
                             classes=self.cat_,
                             cl_labels=cl_labels,
                             **self.plot_kws)

            elif self.plot_mode == 'boxplot':
                cat_boxplot(values=self.cont_,
                            classes=self.cat_,
                            cl_labels=cl_labels,
                            catplot_kws=self.plot_kws)

            else:
                raise ValueError("plot_mode must be one of ['distplot',"
                                 " 'boxplot'], not {}".format(self.self.plot_mode))
