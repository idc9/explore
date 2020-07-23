from sklearn.base import BaseEstimator
from statsmodels.stats.multitest import multipletests
from textwrap import dedent

from explore.utils import has_and_not_none

# TODO: who should own alpha -- test collections or individual tests


class TestMixin(BaseEstimator):
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.plot_kws = {}  # key word arguments for visualization.

    @property
    def rejected_(self):
        """
        Returns true iif the null is rejected.
        """
        if has_and_not_none(self, 'pval_'):
            return self.pval_ < self.alpha

    @property
    def pval_(self):
        """
        Returns the adjusted p-value if there is one, otherwise
        returns the raw p-value.
        """
        if has_and_not_none(self, 'pval_adj_'):
            return self.pval_adj_
        elif has_and_not_none(self, 'pval_raw_'):
            return self.pval_raw_
        else:
            return None

    def _send_pval(self):
        """
        Sends raw pval for correction.

        Output
        ------
        pval: list of flaot (length 1)
            The raw pval for this test wrapped in a list.
        """
        return [self.pval_]  # TODO: should we send pval_ or pval_raw_

    def _receive_pval(self, p):
        """
        Sets adjusted pvalue after correcting for multiple testing.

        Parameters
        ----------
        p: list of float
            The adjusted pval.
        """
        self.pval_adj_ = p[0]

    def plot(self, verbosity=1):
        """
        Visual resresentation/diagnositc for test.
        Sub-class should overwrite.

        Parameters
        ----------
        verbosity: int
            Amount of detail to include in the plot.
        """
        raise NotImplementedError


TestMixin.__doc__ = dedent("""\

    Mixin class for a single test.

    Subclasses should have the following attributes

    pval_raw_: float
        The raw pvalue from the test (before multiple testing adjustment).


    Subclasses should have the following functions

    plot:
        Creates a visual summary/diagnostic of the test results.
    """)


class TestCollectionMixin(BaseEstimator):
    """
    Superclass for a collection of tests.

    Parameters
    ----------
    alpha: float
        Cutoff for significance.

    multi_test: str
        Which procedure to use to control for multiple testing.
        See method argument in statsmodels.stats.multitest.multipletests.

    """
    def __init__(self, alpha=0.05, multi_test='fdr_bh'):
        self.alpha = alpha
        self.multi_test = multi_test

    def _send_pval(self):
        """
        Sends raw pvalues for correction.

        Output
        ------
        pvals: list of floats
            The raw pvals in this collection.
        """
        pvals = []
        for test in self.test_iter():
            pvals += test._send_pval()
        return pvals

    def _receive_pval(self, p):
        """
        Receives corrected pvals and submits them to tests contained in this collection.

        Parameters
        ----------
        p: list of floats
            The corrected pvals for this collection.
        """
        idx = 0
        for test in self.test_iter():
            num_pvals = len(test._send_pval())
            test._receive_pval(p[idx:(idx + num_pvals)])
            idx += num_pvals

    def correct_multi_tests(self):
        """
        Runs multiple testing correction procedure.
        """
        raw_pvals = self._send_pval()

        if len(raw_pvals) == 1:
            pval_corr = raw_pvals

        else:
            rej_corr, pval_corr, _, __ = \
                multipletests(pvals=raw_pvals,
                              alpha=self.alpha,
                              method=self.multi_test)

        self._receive_pval(pval_corr)

        return self

    def test_iter(self):
        """
        Iterates through all tests in this collection. Each iteration should yeild a subclass of either TestCollection or Test.
        Subclasses should overwrite.
        """
        raise NotImplementedError

    def plot(self, verbosity=1):
        """
        Visual resresentations/diagnositcs for tests in this collection.
        Sub-class should overwrite.

        Parameters
        ----------
        verbosity: int
            Amount of detail to include in the plot.
        """
        raise NotImplementedError


_test_collection_docs = dict(
    test_collection_params=dedent("""\
    alpha: float
        Significance level.

    multi_test: str
        Which procedure to use to control for multiple testing.
        See method argument in statsmodels.stats.multitest.multipletests.
    """)
)

TestCollectionMixin.__doc__ = dedent("""\

    Mixin class for a collection single tests.

    Parameters
    ----------
    {test_collection_params}
    """.format(**_test_collection_docs))


class Union(TestCollectionMixin):
    """
    Union of tests.

    Parameters
    ----------
    alpha: float
        Cutoff for significance.

    multi_test: str
        Which procedure to use to control for multiple testing.
        See method argument in statsmodels.stats.multitest.multipletests.
    """
    def __init__(self, alpha=0.05, multi_test='fdr_bh'):
        self.alpha = alpha
        self.multi_test = multi_test
        self.tests_ = {}

    def add_tests(self, tests):
        """
        Add tests to this collection.

        Parameters
        ----------
        tests:
            List of tuples of (name, test)
        """
        for name, test in tests:
            self.add_test(name, test)
        return self

    def add_test(self, name, test):
        """
        Add a single test to this collection..

        Parameters
        ----------
        name: str
            Name of the test.

        test: Test or TestCollection
            The test to add.
        """
        self.tests_[name] = test
        return self

    def test_iter(self):
        """
        Subclasses should overwrite
        """
        for name in self.tests_.keys():
            yield self.tests_[name]
