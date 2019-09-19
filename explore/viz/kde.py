import numpy as np
import warnings
from scipy import stats
from six import string_types
import matplotlib.pyplot as plt
from scipy.integrate import trapz

from explore.utils import Proportions


try:
    import statsmodels.nonparametric.api as smnp
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False


def _univariate_kde(data, shade=False, vertical=False, kernel='gau',
                    bw="scott", gridsize=100, cut=3,
                    clip=None, legend=True, ax=None, cumulative=False,
                    **kwargs):
    """
    Computes the KDE of univariate data.

    shade : bool, optional
        If True, shade in the area under the KDE curve (or draw with filled
        contours when data is bivariate).
    vertical : bool, optional
        If True, density is on x-axis.
    kernel : {'gau' | 'cos' | 'biw' | 'epa' | 'tri' | 'triw' }, optional
        Code for shape of kernel to fit with. Bivariate KDE can only use
        gaussian kernel.
    bw : {'scott' | 'silverman' | scalar | pair of scalars }, optional
        Name of reference method to determine kernel size, scalar factor,
        or scalar for each dimension of the bivariate plot. Note that the
        underlying computational libraries have different interperetations
        for this parameter: ``statsmodels`` uses it directly, but ``scipy``
        treats it as a scaling factor for the standard deviation of the
        data.
    gridsize : int, optional
        Number of discrete points in the evaluation grid.
    cut : scalar, optional
        Draw the estimate to cut * bw from the extreme data points.
    clip : pair of scalars, or pair of pair of scalars, optional
        Lower and upper bounds for datapoints used to fit KDE. Can provide
        a pair of (low, high) bounds for bivariate plots.
    legend : bool, optional
        If True, add a legend or label the axes when possible.
    cumulative : bool, optional
        If True, draw the cumulative distribution estimated by the kde.

    ax : matplotlib axes, optional
        Axes to plot on, otherwise uses current axes.
    kwargs : key, value pairings
        Other keyword arguments are passed to ``plt.plot()`` or
        ``plt.contour{f}`` depending on whether a univariate or bivariate
        plot is being drawn.

    Output
    ------
    x: array-like, (n_grid_points, )
        The grid of values where the kde is evaluated.

    y: array-like, (n_grid_points, )
        The values of the KDE.
    """

    # Sort out the clipping
    if clip is None:
        clip = (-np.inf, np.inf)

    # Calculate the KDE

    if np.nan_to_num(data.var()) == 0:
        # Don't try to compute KDE on singular data
        msg = "Data must have variance to compute a kernel density estimate."
        warnings.warn(msg, UserWarning)
        x, y = np.array([]), np.array([])

    elif _has_statsmodels:
        # Prefer using statsmodels for kernel flexibility
        x, y = _statsmodels_univariate_kde(data, kernel, bw,
                                           gridsize, cut, clip,
                                           cumulative=cumulative)
    else:
        # Fall back to scipy if missing statsmodels
        if kernel != "gau":
            kernel = "gau"
            msg = "Kernel other than `gau` requires statsmodels."
            warnings.warn(msg, UserWarning)
        if cumulative:
            raise ImportError("Cumulative distributions are currently "
                              "only implemented in statsmodels. "
                              "Please install statsmodels.")
        x, y = _scipy_univariate_kde(data, bw, gridsize, cut, clip)

    # Make sure the density is nonnegative
    y = np.amax(np.c_[np.zeros_like(y), y], axis=1)

    return x, y


def _statsmodels_univariate_kde(data, kernel, bw, gridsize, cut, clip,
                                cumulative=False):
    """Compute a univariate kernel density estimate using statsmodels."""
    fft = kernel == "gau"
    kde = smnp.KDEUnivariate(data)
    kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip)
    if cumulative:
        grid, y = kde.support, kde.cdf
    else:
        grid, y = kde.support, kde.density
    return grid, y


def _scipy_univariate_kde(data, bw, gridsize, cut, clip):
    """Compute a univariate kernel density estimate using scipy."""
    try:
        kde = stats.gaussian_kde(data, bw_method=bw)
    except TypeError:
        kde = stats.gaussian_kde(data)
        if bw != "scott":  # scipy default
            msg = ("Ignoring bandwidth choice, "
                   "please upgrade scipy to use a different bandwidth.")
            warnings.warn(msg, UserWarning)
    if isinstance(bw, string_types):
        bw = "scotts" if bw == "scott" else bw
        bw = getattr(kde, "%s_factor" % bw)() * np.std(data)
    grid = _kde_support(data, bw, gridsize, cut, clip)
    y = kde(grid)
    return grid, y


def _kde_support(data, bw, gridsize='default', cut=3, clip=None):
    """Establish support for a kernel density estimate."""
    support_min = max(data.min() - bw * cut, clip[0])
    support_max = min(data.max() + bw * cut, clip[1])
    return np.linspace(support_min, support_max, gridsize)


def get_class_kdes(values, classes, ensure_norm=True, **kde_kws):
    """
    KDEs for values with associated classes. Computes the KDE of each class
    then weights each KDE by the number of points in each class. Also
    compute the overall KDE.

    Output
    ------
    cl_kdes, overall_kde

    cl_kdes: dict
        KDE for each class. Keys are class labels.

    overall_kde: dict
        Overall KDE (i.e. ignoring class labels)
    """

    # TODO: do we really need ensure_norm

    overall_grid, overall_y = _univariate_kde(values, **kde_kws)
    if ensure_norm:
        overall_y = norm_kde(grid=overall_grid, y=overall_y)
    overall_kde = {'grid': overall_grid, 'y': overall_y}

    cl_props = Proportions(classes)
    cl_kdes = {}
    for cl in np.unique(classes):
        cl_mask = classes == cl
        cl_values = values[cl_mask]

        cl_grid, cl_y = _univariate_kde(cl_values, **kde_kws)

        if ensure_norm:
            cl_y = norm_kde(grid=cl_grid, y=cl_y)

        # weight area under KDE by number of samples
        cl_y *= cl_props[cl]
        cl_kdes[cl] = {'grid': cl_grid,
                       'y': cl_y}

    return cl_kdes, overall_kde


def norm_kde(grid, y):
    tot = trapz(y=y, x=grid)
    return y / tot


def _univariate_kdeplot(x, y, shade=True, vertical=False,
                        legend=True, ax=None, **kwargs):
    """Plot a univariate kernel density estimate on one of the axes."""

    if ax is None:
        ax = plt.gca()

    # Make sure the density is nonnegative
    y = np.amax(np.c_[np.zeros_like(y), y], axis=1)

    # Flip the data if the plot should be on the y axis
    if vertical:
        x, y = y, x

    # Check if a label was specified in the call
    label = kwargs.pop("label", None)

    # Otherwise check if the data object has a name
    if label is None and hasattr(x, "name"):
        label = x.name

    # Decide if we're going to add a legend
    legend = label is not None and legend
    label = "_nolegend_" if label is None else label

    # Use the active color cycle to find the plot color
    facecolor = kwargs.pop("facecolor", None)
    line, = ax.plot(x, y, **kwargs)
    color = line.get_color()
    line.remove()
    kwargs.pop("color", None)
    facecolor = color if facecolor is None else facecolor

    # Draw the KDE plot and, optionally, shade
    ax.plot(x, y, color=color, label=label, **kwargs)
    shade_kws = dict(
        facecolor=facecolor,
        alpha=kwargs.get("alpha", 0.25),
        clip_on=kwargs.get("clip_on", True),
        zorder=kwargs.get("zorder", 1),
        )
    if shade:
        if vertical:
            ax.fill_betweenx(y, 0, x, **shade_kws)
        else:
            ax.fill_between(x, 0, y, **shade_kws)

    # Set the density axis minimum to 0
    if vertical:
        ax.set_xlim(0, auto=None)
    else:
        ax.set_ylim(0, auto=None)

    # Draw the legend here
    handles, labels = ax.get_legend_handles_labels()
    if legend and handles:
        ax.legend(loc="best")

    return ax


def _univariate_conditional_kdeplot(values, classes,
                                    cl_labels=None,
                                    cl_palette=None,
                                    include_overall=True,
                                    shade=True,
                                    vertical=False,
                                    legend=True,
                                    ax=None,
                                    kde_kws={},
                                    kde_plt_kws={}):

    cl_kdes, overall_kde = get_class_kdes(values, classes, **kde_kws)

    # in case 'overall' is one of the classes
    if 'overall' in np.unique(classes):
        overall_name = ''.join(np.unique(classes))
    else:
        overall_name = 'overall'
    cl_kdes[overall_name] = overall_kde

    # plot the KDE for each class
    for cl in cl_kdes.keys():
        _kwargs = kde_plt_kws.copy()
        _kwargs['shade'] = shade

        x = cl_kdes[cl]['grid']
        y = cl_kdes[cl]['y']

        if cl_palette is not None and cl in cl_palette:
            _kwargs['color'] = cl_palette[cl]

        if cl_labels is not None and cl in cl_labels:
            _kwargs['label'] = cl_labels[cl]
        else:
            _kwargs['label'] = cl

        if cl == overall_name:
            if not include_overall:
                continue

            # _kwargs['alpha'] = .2
            _kwargs['zorder'] = 1
            _kwargs['label'] = 'overall'
            _kwargs['color'] = 'gray'
            _kwargs['shade'] = False

        _univariate_kdeplot(x=x, y=y,
                            vertical=vertical,
                            legend=legend, ax=ax, **_kwargs)
