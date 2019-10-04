import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from numbers import Number
from collections import Counter
import pandas as pd
import seaborn as sns


class ABLine2D(plt.Line2D):

    """
    Draw a line based on its slope and y-intercept. Additional arguments are
    passed to the <matplotlib.lines.Line2D> constructor.

    from https://stackoverflow.com/questions/7941226/how-to-add-line-based-on-slope-and-intercept-in-matplotlib
    """

    def __init__(self, slope, intercept, *args, **kwargs):

        # get current axes if user has not specified them
        if'axes' not in kwargs.keys():
            kwargs.update({'axes': plt.gca()})
        ax = kwargs['axes']

#         # if unspecified, get the current line color from the axes
#         if not ('color' in kwargs or 'c' in kwargs):
#             kwargs.update({'color':ax._get_lines.color_cycle.next()})

        # init the line, add it to the axes
        super(ABLine2D, self).__init__([], [], *args, **kwargs)
        self._slope = slope
        self._intercept = intercept
        ax.add_line(self)

        # cache the renderer, draw the line for the first time
        ax.figure.canvas.draw()
        self._update_lim(None)

        # connect to axis callbacks
        self.axes.callbacks.connect('xlim_changed', self._update_lim)
        self.axes.callbacks.connect('ylim_changed', self._update_lim)

    def _update_lim(self, event):
        """ called whenever axis x/y limits change """
        x = np.array(self.axes.get_xbound())
        y = (self._slope * x) + self._intercept
        self.set_data(x, y)
        self.axes.draw_artist(self)


def component_plot(values, zero_index=False,
                   color='black', plot_kws={}, scatter_kws={},
                   lines=True, points=True,
                   nticks=10):
    """
    Plots component index vs values.

    Parameters
    ----------
    values: array-like

    zero_index: bool
        First index value is zero.

    color: str
        Default color for points and lines.

    plot_kws: dict
        Key word arguments for plt.plot

    scatter_kws: dict
        Key word arguments for plt.scatter_kws

    lines: bool
        Include lines.

    points: bool
        Include points.

    nticks: int, 'all'
        Number of xtick intervals. If 'all', will put all ticks
    """

    assert isinstance(nticks, int) or nticks == 'all'
    n_components = len(values)

    if hasattr(values, 'name'):
        ylab = values.name
    else:
        ylab = None

    if zero_index:
        idxs = idxs = np.arange(n_components)
    else:
        idxs = np.arange(1, n_components + 1)

    default_plot_kws = {'color': color}
    default_scatter_kws = {'color': color}

    if len(plot_kws) > 0:
        for k in plot_kws.keys():
            default_plot_kws[k] = plot_kws[k]

    if len(scatter_kws) > 0:
        for k in scatter_kws.keys():
            default_scatter_kws[k] = scatter_kws[k]

    # plot points and lines
    if lines:
        plt.plot(idxs, values, **default_plot_kws)

    if points:
        plt.scatter(idxs, values, **default_scatter_kws)
    plt.xlim(0)

    if ylab:
        plt.ylabel(ylab)

    if nticks == 'all':
        plt.xticks(idxs)
    else:
        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nticks, integer=True))


def add_xticks(ticks):
    if isinstance(ticks, Number):
        ticks = [ticks]
    plt.xticks(list(plt.xticks()[0]) + ticks)


def add_yticks(ticks):
    if isinstance(ticks, Number):
        ticks = [ticks]
    plt.yticks(list(plt.yticks()[0]) + ticks)


def bold(text):
    """
    Makes text blod
    """
    # TODO: this is causing a strange latex error, figure out
    # TODO: escape under scores
    # return r"$\bf{" + text + "}$"
    return text


def count_plot(values, display='both'):
    """
    Plots a horizontal histogram counting the number of samples in each category.

    Parametres
    ----------
    values: array-like, pd.Series, shape (n_sample, )
        The observed values to count. If pd.Series with a name, will use
        name in the legend.

    display: str, ['count', 'prop', 'both']
        Plot counts, proportions or both.
    """
    assert display in ['count', 'prop', 'both']

    # compute counts and proportions
    counts = pd.Series(Counter(values))
    counts = counts.sort_index(ascending=False)  # col ordering

    props = counts / counts.sum()

    # formatting
    if hasattr(values, 'name'):
        cat_name = values.name
    else:
        cat_name = ''

    if display in ['count', 'both']:
        counts.plot.barh(color='black')
    else:
        props.plot.barh(color='black')

    ax = plt.gca()
    for i, (c, p) in enumerate(zip(counts, props)):
        if display == 'count':
            txt = '{:d}'.format(c)
            val = c
        elif display == 'prop':
            txt = '{:1.2f}'.format(p)
            val = p
        else:
            txt = '{:d} ({:1.2f})'.format(c, p)
            val = c

        ax.text(.8 * val, i, txt,
                color='red', fontsize=15, fontweight='bold')

    plt.ylabel(cat_name)
    plt.xlabel('count')


def heatmap(data, **kws):
    """
    Seaborn heatmap without cutting top/bottom off.
    """
    sns.heatmap(data, **kws)
    plt.ylim(data.shape[1], 0)


heatmap.__doc__ = 'Fixes seaborn heatmap issue where top/bottom '\
                  'axes get cutoff (see https://github.com/mwaskom/seaborn/issues/1773) \n {}'.format(sns.heatmap.__doc__)


def fmt_pval(pval):
    """
    Formats a p-value as a string. Use scienfitic notation if p < 0.01.
    """
    if np.allclose(pval, 0):
        return '0.00'
    elif pval < .01:
        return '{:1.2e}'.format(pval)
    else:
        return '{:1.2f}'.format(pval)
