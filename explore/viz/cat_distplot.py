import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import seaborn as sns

from explore.viz.jitter import jitter_yvals
from explore.viz.kde import _univariate_conditional_kdeplot


def cat_distplot(values, classes,
                 cl_labels=None,
                 palette='Set2',
                 points='order', yprops=(.2, .5),
                 jitter_kws={'s': 10},
                 kde_kws={},
                 kde_plt_kws={},
                 shade=True,
                 vertical=False,
                 ax=None):
    """
    Plots category distribution plot i.e. a set of points with
    associated class labels.

    Parameters
    ----------
    values: array-like, shape (n_samples, )
        The set of continuous values.

    classes: array-like, shape (n_samples, )
        The class labels.

    cl_labels: None, dict
        Label for each category. If None, default labeling is used.
        If dict is provided, its keys must correspond to the categories.

    palette: str, dict
        Color palette for classes. Must be either a dict mapping classes
        to colors or a seaborn color palette.

    points: ['cdf', 'random', None]
        Whether or not to add jittering points.

    yprops: (float, float)
        Limit scaling for jitter.

    jitter_kwars: dict
        key word arguments for plt.scatter

    distplot_kwargs: dict
        key word arguments for sns.distplot
    """
    # extract metadata
    if hasattr(classes, 'name'):
        legend_title = classes.name
    else:
        legend_title = ''

    if hasattr(values, 'name'):
        xlab = values.name
    else:
        xlab = ''

    labels = np.unique(classes)
    cl_counts = Counter(classes)

    # setup class labels
    if cl_labels is not None:
        if set(list(cl_labels.keys())) != set(labels):
            raise ValueError('cl_labels keys must correspond to class labels.')
    else:
        cl_labels = {cl: _default_label(cl, cl_counts[cl]) for cl in labels}

    # setup color palette for each class
    cl_palette = get_palette_dict(labels=labels, palette=palette)

    # plot kde
    _univariate_conditional_kdeplot(values, classes,
                                    cl_labels=cl_labels,
                                    cl_palette=cl_palette,
                                    shade=shade,
                                    vertical=vertical,
                                    legend=True,
                                    ax=ax,
                                    kde_kws=kde_kws,
                                    kde_plt_kws=kde_plt_kws)

    # plot jitter points
    if points is not None:
        yvals = jitter_yvals(values, how=points,
                             yprops=yprops,
                             jitter_cdf_width=.1)
        for cl in labels:
            cl_values = values[classes == cl]
            cl_yvals = yvals[classes == cl]

            plt.scatter(cl_values, cl_yvals,
                        color=cl_palette[cl],
                        zorder=len(labels) + 1,
                        **jitter_kws)

    plt.ylim(0)
    plt.legend(title=legend_title, frameon=False, loc='best')
    plt.xlabel(xlab)
    # plt.ylabel(ylab)


def cat_boxplot(values, classes, cl_labels=None, palette='Set2',
                catplot_kws={}):
    """
    Plots a boxplot for each category.

    TODO: document better
    """
    if hasattr(classes, 'name'):
        class_lab = classes.name
    else:
        class_lab = 'classes'

    if hasattr(values, 'name'):
        value_lab = values.name
    else:
        value_lab = 'values'

    labels = np.unique(classes)
    cl_counts = Counter(classes)

    # setup class labels
    if cl_labels is not None:
        if set(list(cl_labels.keys())) != set(labels):
            raise ValueError('cl_labels keys must correspond to class labels.')
    else:
        cl_labels = {cl: _default_label(cl, cl_counts[cl]) for cl in labels}

    # setup color palette for each class
    cl_palette = get_palette_dict(labels=labels, palette=palette)

    df = pd.DataFrame({class_lab: classes,
                       value_lab: values})

    sns.catplot(x=value_lab, y=class_lab, data=df, kind='box', orient='h',
                palette=cl_palette, **catplot_kws)

    for cl in labels:
        med = np.median(values[classes == cl])
        plt.axvline(med, color=cl_palette[cl], label=cl_labels[cl])

    plt.legend(title=class_lab, frameon=False, loc='best')


def _default_label(cl, cl_count):
    return '{} (n = {})'.format(cl, cl_count)


def get_palette_dict(labels, palette='Set2'):
    """
    Returns a color palette as a dict keyed by the unique entries of labels.
    """
    labels = np.unique(labels)
    if type(palette) == str:
        color_pal = sns.color_palette(palette, len(labels))
        cl2col = {cl: color_pal[i] for i, cl in enumerate(labels)}
    elif type(palette) == dict:
        cl2col = palette
    else:
        raise ValueError('palette must be either str or dict, not {}'.format(palette))
    return cl2col
