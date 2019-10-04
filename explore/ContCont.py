from scipy.stats import pearsonr, spearmanr

from explore.Base import TestMixin
from explore.utils import process_var_pair, is_cat
from explore.viz.continuous import get_cts_label, plot_scatter
from explore.viz.utils import bold, fmt_pval


class ContCont(TestMixin):
    """
    Continuous vs. continuous variable comparison.

    Parameters
    ----------
    alpha: float
        Significance level.

    measure: str
        Correlation measure to use, must be one of ['pearson', 'spearman'].

    Attributes
    ----------
    corr_: float
        The correlation value between the two variables.

    pval_: float
        The pval for test of nonzero correlation.

    rejected_: bool
        If null of zero correlation is rejected.

    plot_kws: dict
        Key word arguments for explore.viz.continuous.plot_scatter
    """

    def __init__(self, alpha=0.05, measure='pearson'):
        self.alpha = alpha
        self.measure = measure

        self.plot_kws = {}

    def fit(self, a, b):
        """
        Parameters
        ----------
        a: array-like, (n_samples, )
            Continuous variable.

        b: array-like, (n_samples, )
            Continuous variable.
        """
        if is_cat(a):
            raise ValueError('a must be a continuous variable.')
        if is_cat(b):
            raise ValueError('b must be a continuous variable.')

        # processing
        self.a_, self.a_nan_idxs_, self.b_, self.b_nan_idxs_ = \
            process_var_pair(a, b, nan_how_a='drop', nan_how_b='drop')
        self.var_names_ = [self.a_.name, self.b_.name]

        # compute correlation and significance test
        if self.measure == 'pearson':
            corr, pval = pearsonr(self.a_, self.b_)

        elif self.measure == 'spearman':
            spear = spearmanr(self.a_, self.b_)
            corr = spear.correlation
            pval = spear.pvalue

        elif callable(self.measure):
            corr, pval = self.measure(self.a_, self.b_)

        else:
            raise ValueError('measure must be in ["pearson", "spearman"]')

        self.corr_ = corr  # correlation value
        self.pval_raw_ = pval

        return self

    def text_summary(self, how='full'):

        names_txt = '{} vs. {}'.format(self.var_names_[0], self.var_names_[1])
        quant_summary = '{} = {:1.3f} (p={})'.format(self.measure,
                                                     self.corr_,
                                                     fmt_pval(self.pval_))
        if self.rejected_:
            quant_summary += '*'
            quant_summary = bold(quant_summary)

        if how == 'quant_only':
            return quant_summary
        elif how == 'full':
            return '{}\n{}'.format(names_txt, quant_summary)
        else:
            raise ValueError('how must be one of ["full", "quant_only"], not {}'.format(how))

    def plot(self, verbosity=1):
        """
        Plots a scatter plot with OLS regression line.

        Parameters
        ----------
        verbosity: int
            Amount of detail to include in the plot.
        """
        if callable(self.measure):
            corr_name = self.measure.__name__
        else:
            corr_name = self.measure

        label = get_cts_label(reject=self.rejected_,
                              corr=self.corr_,
                              corr_name=corr_name,
                              pval=self.pval_)

        plot_scatter(x=self.a_, y=self.b_,
                     alpha=self.alpha,
                     label=label,
                     **self.plot_kws)
