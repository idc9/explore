"""
Univariate, binary class difference in distribution tests.
"""
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp, anderson_ksamp

from explore.viz.utils import bold


def binary_dist_test(a, b, test='auc'):
    """
    Wrapper for difference of distribution tests for univariate observations
    from two classes.

    Parameteres
    -----------
    a, b: array-like
        The observation values in each class.

    test: str (['ad','mw', 'ks', 't']), callable
        Which test to use.
        'ad': Anderson-Darling (general test for differing distributions)
        'ks': Kolmogorov-Smirnov (general test for differing distributions)
        't': t-test for difference in locations.
        'mw': Mann Whitney U test for difference in locations (reports AUC staistic)

        if callable, should take two array-like arguments.

    Output
    ------
    stat, pval

    stat: float
        The test statistic such that larger values mean bigger differences.

    pval: float
        The p value.

    """

    if test == 't':
        stat, pval = ttest_ind(a, b, equal_var=False)
        stat = abs(stat)

    elif test in ['auc', 'mw', 'mannwhitneyu']:

        result = binary_mann_whitney_u(a, b)
        pval = result['pval']  # makes two-sided
        stat = result['auc']

    elif test == 'ks':
        stat, pval = ks_2samp(a, b)

    elif test == 'ad':
        stat, _, pval = anderson_ksamp([a, b])

    elif callable(test):
        stat, pval = test(a, b)

    else:
        raise ValueError('test = {} is is not acceptable value.'.format(test))

    return stat, pval


def binary_mann_whitney_u(a, b, alternative='two-sided', method='auto',):
    """
    Mann- Whitney U test for difference in locations. Wraps scipy.stats.mannwhitneyu. Reports the larger U statisic. Also
    compute 'auc' and 'z' statistics.

    Note the Mann-Whitney U statistics can be rescaled to be equal to the AUC
    scores.

    Parameters
    ----------
    a: array-like, shape (n_samples, )
        The observation values in the first class.

    b: array-like, shape (n_samples, )
        The observation values in the second class.

    alternative: str
        The alternative hypotheses; must be one of ['two-sided', 'less', 'greater']. See scipy.stats.mannwhitneyu.

    method: str
        How to compute the p-value; must be one of ['auto', 'exact']. See scipy.stats.mannwhitneyu.

    Output
    ------
    results: dict

    results['U']: float
        The U statisitic

    results['Z']: float
        The Z statisitc.

    results['auc']: float
        The AUC statisitic.

    results['pval']: float
        The pvalue.
    """

    results = mannwhitneyu(a, b, use_continuity=True,
                           alternative=alternative, method=method)
    U = results.statistic

    # let's report the larger U instead
    U = len(a) * len(b) - U

    # compute AUC
    auc = U / (len(a) * len(b))

    # compute Z
    m = len(a) * len(b) / 2
    sigma = np.sqrt(len(a) * len(b) * (len(a) + len(b) + 1) / 12)
    z = (U - m) / sigma

    if alternative == 'two-sided':
        z = abs(z)

    return {'U': U,
            'auc': auc,
            'Z': z,
            'pval': results.pvalue}


def _get_ovr_label(cl, n, stat_name, stat, reject, test_prefix='one-vs-rest'):

    if reject:
        label = r"$\bf{" + cl + "}$" + '*'
    else:
        label = cl

    label += ' (n = {}, '.format(n)
    if test_prefix is not None:
        label += str(test_prefix)

    label += ' {} = {:1.2f})'.format(stat_name, stat)

    # label += ' (n = {}, one-vs-rest {} = {:1.2f})'.format(n, stat_name, stat)

    return label


def _get_ovo_labels(ovo_tests, summary='list'):
    """
    Gets labels for one-vs-one tests using output of get_ovo_tests()
    """
    test_name = ovo_tests.test

    cl_labels = {}
    for current_cl in ovo_tests.labels_:

        results = ovo_tests.get_cl_ovo_results(current_cl)

        cl_counts = ovo_tests.counts_[current_cl]

        # for the current class see which other classes are separated
        other_rejected = results.index[results['rejected']]

        if len(other_rejected) > 0:

            # label = r"$\bf{" + current_cl + "}$"
            label = bold(current_cl) + '*'
            label += ', n = {}, one-vs-one {} test'.format(cl_counts,
                                                           test_name)

            if summary == 'list':
                label += ': '
                # label += '\n'
                for other_cl in other_rejected:
                    stat = results.loc[other_cl, 'stat']
                    # label += other_cl
                    # label += '{} ({} = {:1.3f})'.format(other_cl, test, rej[other_cl])
                    label += '{} ({:1.3f}) '.format(other_cl, stat)

            elif summary == 'avg':

                avg = np.mean(results['stat'])
                # avg = np.mean(stats.loc[cl, rej_classes].drop(index=cl)) # TODO: which one
                label += ' (avg = {:1.3f})'.format(avg)

            elif summary == 'max':
                max_val = np.max(results['stat'])
                label += ' (max = {:1.3f})'.format(max_val)

        else:
            label = '{}, n = {}, {} test'.format(current_cl,
                                                 cl_counts,
                                                 test_name)

        cl_labels[current_cl] = label

    return cl_labels
