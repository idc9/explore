import numpy as np
from textwrap import dedent

from explore.utils import is_cat
from explore.CatCat import CatCat
from explore.ContCont import ContCont
from explore.ContCat import ContCat
from explore.ContBinCat import ContBinCat
from explore.Base import TestCollectionMixin, _test_collection_docs


class PairwiseTestsMixin(TestCollectionMixin):

    def __init__(self, alpha=0.05, multi_test='fdr_bh',
                 corr='pearson', cat_test='auc',
                 multi_cat='ovo', nan_how='drop'):

        self.alpha = alpha
        self.multi_test = multi_test
        self.corr = corr
        self.cat_test = cat_test
        self.multi_cat = multi_cat
        self.nan_how = nan_how

    def _get_test(self, a, b):
        """
        Gets the test to compare a to b. Infers whether a and b are
        categorical or continuous and selects the appropriate test.

        Parameters
        ----------
        a, b: array-like, (n_samples, )
            The two lists of observations.
        """
        if is_cat(a) and is_cat(b):
            tst = CatCat(alpha=self.alpha, nan_how=self.nan_how)
            tst.fit(a, b)

        elif (not is_cat(a)) and (not is_cat(b)):
            tst = ContCont(alpha=self.alpha)
            tst.fit(a, b)

        elif (not is_cat(a)) and is_cat(b):
            if len(np.unique(b)) != 2:
                tst = ContCat(alpha=self.alpha,
                              test=self.cat_test,
                              multi_cat=self.multi_cat,
                              nan_how=self.nan_how)
                tst.fit(a, b)
            else:
                # TODO: maybe modify ContCat to return ContBinCat if two classes
                tst = ContBinCat(alpha=self.alpha,
                                 test=self.cat_test)
                tst.fit(a, b)

        elif is_cat(a) and (not is_cat(b)):
            if len(np.unique(b)) != 2:
                tst = ContCat(alpha=self.alpha,
                              test=self.cat_test,
                              multi_cat=self.multi_cat,
                              nan_how=self.nan_how)
                tst.fit(b, a)
            else:
                tst = ContBinCat(alpha=self.alpha,
                                 test=self.cat_test)
                tst.fit(b, a)

        return tst


_pairwise_docs = dict(
    pairwise_params=dedent("""\
    {test_collection_params}

    corr: str
        Correlation measure to use, must be one of ['pearson', 'spearman'] or callable with attribute name.

    cat_test: str
        Which difference in distribution test to use, see test argument in
        explore.univariate_binary_dists_tests.binary_dist_test.

    multi_cat: str
        Which pairwise comparisons to make for multiple categories (one-vs-one or one-vs-rest). Must be one of ['ovo', 'ovr'].

    nan_how: str
        How to handle nan values in the categorical variable. Must be one
        of ['drop', 'leave']. If 'leave', nan values are treated as a category.
        Note continuous nan values are always dropped.\
        """.format(**_test_collection_docs))

)

PairwiseTestsMixin.__doc__ = dedent("""\
    Mixin class for comparing all paris of variables between two sets of variables.

    Parameters
    ----------
    {pairwise_params}

    """).format(**_pairwise_docs)
