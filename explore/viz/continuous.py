import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from explore.utils import safe_apply
from explore.viz.utils import bold, ABLine2D, fmt_pval


def plot_scatter(x, y, alpha=0.05, standardize=False, label=None):
    """
    Parameters
    ----------
    x, y: array-like (ideally pd.Series)
        x, y values to plot. If pd.Series, uses 'name' to get x/y labels

    alpha: float
        Cutoff for correlation coefficient significance.

    standardisze: bool
        Whether or not to standardized (mean center and scale) variables.
        True by defualt.

    """

    xlab, ylab = '', ''
    if hasattr(x, 'name'):
        xlab = x.name
    if hasattr(y, 'name'):
        ylab = y.name

    # drop missing values
    df = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()

    # optinally center/scale
    if standardize:
        df = safe_apply(StandardScaler().fit_transform, df)
        xlab += ' (standardized)'
        ylab += ' (standardized)'

    x = df.iloc[:, 0].values.reshape(-1)
    y = df.iloc[:, 1].values.reshape(-1)

    # fit linear model
    lm = LinearRegression(fit_intercept=True).fit(x.reshape(-1, 1), y)
    slope = lm.coef_.item()
    intercept = lm.intercept_

    # if no label provided, compute correlation
    if label is None:
        alpha = 0.05
        # compute pearson correlation
        corr, pval = pearsonr(x, y)
        reject = pval < alpha
        label = get_cts_label(reject, corr, corr_name='pearson', pval=pval)

    # scatter plot
    plt.scatter(x, y, color='blue', s=2)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

    # line
    ABLine2D(slope, intercept, label=label,
             color='blue')  # , linewidth=linewidth
    plt.legend(loc='upper left')


def get_cts_label(reject, corr, corr_name, pval):

    if reject:
        # stat_str = bold('pearson \\ corr: {:1.2f} \\ (p={:1.2f})'.format(corr, pval))
        # label = bold('{}: {:1.3f} (p={:1.3f})*'.format(corr_name, corr, pval))
        # label = bold('{}: {:1.3f} (p={:.1e})*'.format(corr_name, corr, pval))
        label = bold('{}: {:1.3f} (p={})*'.format(corr_name, corr,
                                                  fmt_pval(pval)))
    else:
        # stat_str = 'pearson corr: {:1.2f} (p={:1.2f})'.format(corr, pval)
        # label = '{}: {:1.3f} (p={:1.3f})'.format(corr_name, corr, pval)
        label = '{}: {:1.3f} (p={})'.format(corr_name, corr,
                                            fmt_pval(pval))

    return label
