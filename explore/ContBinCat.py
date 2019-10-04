import numpy as np

from explore.Base import TestMixin
from explore.viz.cat_distplot import cat_distplot, cat_boxplot
from explore.ubdt import binary_dist_test, _get_ovr_label
from explore.utils import process_var_pair, is_cat, get_counts


class ContBinCat(TestMixin):
    """
    Continuous vs. binary categorical variable comparison.

    Parameters
    ----------
    alpha: float
        Significance level.

    test: str
        Which difference in distribution test to use, see
        swdb.compare.univariate_binary_dists_tests.binary_dist_test.

    multi_cat: str
        Which pairwise comparisons to make for multiple categories (one-vs-one or one-vs-rest). Must be one of ['ovo', 'ovr'].

    Attributes
    ----------
    stat_

    pval_

    """

    def __init__(self, alpha=0.05, test='auc'):
        self.alpha = alpha
        self.test = test

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
            raise ValueError('cont must be a continuous variable.')
        if not is_cat(cat):
            raise ValueError('cat must be a categorical variable.')

        # process intpu
        self.cont_, self.cont_nan_idxs_, self.cat_, self.cat_nan_idxs_ = \
            process_var_pair(cont, cat,
                             nan_how_a='drop',
                             nan_how_b='drop')
        self.var_names_ = [self.cont_.name, self.cat_.name]

        self.cat_ = self.cat_.astype(str)
        self.labels_ = np.unique(self.cat_)
        self.counts_ = get_counts(self.cat_)
        self.n_cats_ = len(self.labels_)

        if self.n_cats_ != 2:
            raise ValueError('y must have 2 classes, not {}'.
                             format(len(self.labels_)))

        # split the class 0/1 observations into two vectors
        cl_0 = self.cont_[self.cat_ == self.labels_[0]]
        cl_1 = self.cont_[self.cat_ == self.labels_[1]]
        # cl_0 = self.cont_[nansafe_equal_mask(self.cat_, self.labels_[0])]
        # cl_1 = self.cont_[nansafe_equal_mask(self.cat_, self.labels_[1])]

        self.stat_, self.pval_raw_ = binary_dist_test(cl_0, cl_1,
                                                      test=self.test)

        return self

    # def text_summary(self, how='full'):

    #     names_txt = '{} vs. {}'.format(self.var_names_[0], self.var_names_[1])

    #     quant_summary = '{} = {:1.3f} (p={:1.3f})'.format(self.test,
    #                                                       self.stat_,
    #                                                       self.pval_)
    #     if self.rejected_:
    #         quant_summary += '*'
    #         quant_summary = bold(quant_summary)

    #     if how == 'quant_only':
    #         return quant_summary
    #     elif how == 'full':
    #         return '{}\n{}'.format(names_txt, quant_summary)
    #     else:
    #         raise ValueError('how must be one of ["full", "quant_only"], not {}'.format(how))

    def plot(self, verbosity=1):
        """
        Parameters
        ----------
        verbosity: int
            Amount of detail to include in the plot.
        """

        cl_labels = {}
        for cl in self.labels_:

            if verbosity >= 1:
                label_kws = {'cl': cl,
                             'n': self.counts_[cl],
                             'stat_name': self.test,
                             'stat': self.stat_,
                             'reject': self.rejected_,
                             'test_prefix': ''}

                cl_labels[cl] = _get_ovr_label(**label_kws)
            else:
                cl_labels[cl] = cl

        # cat_distplot(values=self.cont_,
        #              classes=self.cat_,
        #              cl_labels=cl_labels)

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
