"""
Visualization module.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import functools
import logging
from typing import Callable, List, Literal, Dict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors
from matplotlib.colors import LogNorm
from matplotlib.container import BarContainer
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import Parametrization
from .spectrum import Spectrum

# get logger
logger = logging.getLogger('fastdfe')


class Visualization:
    """
    Visualization class.
    """

    # configure color map
    # plt.rcParams['axes.prop_cycle'] = cycler('color', plt.get_cmap('Set2').colors)

    @classmethod
    def change_default_figsize(cls, factor: float | np.ndarray):
        """
        Scale default figure size.

        :return: Factor to scale default figure size by
        """
        plt.rcParams["figure.figsize"] = list(factor * np.array(plt.rcParams["figure.figsize"]))

    @staticmethod
    def clear_show_save(func: Callable) -> Callable:
        """
        Decorator for clearing current figure in the beginning
        and showing or saving produced plot subsequently.

        :param func: Function to decorate
        :return: Wrapper function
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> plt.Axes:
            """
            Wrapper function.

            :param args: Positional arguments
            :param kwargs: Keyword arguments
            :return: Axes
            """

            # add axes if not given
            if 'ax' not in kwargs or ('ax' in kwargs and kwargs['ax'] is None):
                # clear current figure
                plt.clf()

                kwargs['ax'] = plt.gca()

            # execute function
            func(*args, **kwargs)

            # make layout tight
            plt.tight_layout()

            # show or save
            # show by default here
            return Visualization.show_and_save(
                file=kwargs['file'] if 'file' in kwargs else None,
                show=kwargs['show'] if 'show' in kwargs else True
            )

        return wrapper

    @staticmethod
    def show_and_save(file: str = None, show: bool = True) -> plt.Axes:
        """
        Show and save plot.

        :param file: File path to save plot to
        :param show: Whether to show plot
        :return: Axes

        """
        # save figure if file path given
        if file is not None:
            plt.savefig(file, dpi=200, bbox_inches='tight', pad_inches=0.1)

        # show figure if specified and if not in interactive mode
        if show and not plt.isinteractive():
            plt.show()

        # return current axes
        return plt.gca()

    @staticmethod
    def interval_to_string(left: float, right: float) -> str:
        """
        Get string representation for given interval.

        :param left: Right interval boundary
        :param right: Left interval boundary
        :return: String representation of interval

        """
        # left interval is closed by default
        bracket_left = '[' if left != -np.inf else '('
        bracket_right = ')'

        def format_number(n: float) -> str:
            """
            Format number, allowing for np.inf.

            :param n: Number to format
            :return: Formatted number
            """
            if np.abs(n) != np.inf:
                return '{:0.0f}'.format(n)

            return str(np.inf)

        return bracket_left + format_number(left) + ', ' + format_number(right) + bracket_right

    @staticmethod
    @clear_show_save
    def plot_discretized(
            ax: plt.Axes,
            values: list | np.ndarray,
            errors: list | np.ndarray = None,
            labels: list | np.ndarray = None,
            file: str = None,
            show: bool = True,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            title: str = 'discretized DFE',
            interval_labels: List[str] = None,

    ) -> plt.Axes:
        """
        Plot discretized DFEs using a bar plot.

        :param interval_labels: Labels for the intervals
        :param labels: Labels for the DFEs
        :param title: Title of the plot
        :param values: Array of values of size ``intervals.shape[0] - 1``
        :param errors: Array of errors of size ``intervals.shape[0] - 1``
        :param file: File path to save plot to
        :param show: Whether to show plot
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bars.
        :param ax: Axes to plot on
        :return: Axes
        """
        # number of intervals
        n_intervals = len(intervals) - 1
        n_dfes = len(values)

        width_total = 0.9
        width = width_total / n_dfes

        for i in range(n_dfes):
            x = np.arange(n_intervals) + i * width

            # plot discretized DFE
            bars = ax.bar(
                x=x,
                height=values[i],
                width=width,
                yerr=errors[i],
                error_kw=dict(
                    capsize=width * 7
                ),
                label=labels[i] if labels is not None else None,
                linewidth=0,
                hatch=Visualization.get_hatch(i, labels),
                color=Visualization.get_color(labels[i], labels) if labels is not None else None
            )

            Visualization.darken_edge_colors(bars)

        ax.set(xlabel='S', ylabel='fraction')

        # determine x-labels
        if interval_labels is None:
            xlabels = [Visualization.interval_to_string(intervals[i - 1], intervals[i]) for
                       i in range(1, n_intervals + 1)]
        else:
            xlabels = interval_labels

        # customize x-ticks
        x = np.arange(n_intervals)
        ax.set_xticks([i + (width_total - width) / 2 for i in x], x)
        ax.set_xticklabels(xlabels)

        # set title
        ax.set_title(title)

        # show legend if labels were given
        if labels is not None:
            plt.legend(prop=dict(size=8))

        # remove x-margins
        ax.autoscale(tight=True, axis='x')

        return ax

    @staticmethod
    @clear_show_save
    def plot_continuous(
            ax: plt.Axes,
            bins: np.ndarray,
            values: list | np.ndarray,
            errors: list | np.ndarray = None,
            labels: list | np.ndarray = None,
            file: str = None,
            show: bool = True,
            title: str = 'continuous DFE',
            scale: Literal['log', 'linear'] = 'lin',
            ylim: float = 1e-2,
            scale_density: bool = False,
            **kwargs
    ) -> plt.Axes:
        """
        Plot DFEs using a line plot.
        By default, the PDF is plotted as is. Due to the logarithmic scale on
        the x-axis, we may get a wrong intuition on how the mass is distributed,
        however. To get a better intuition, we can optionally scale the density
        by the x-axis interval size using ``scale_density = True``. This has the
        disadvantage that the density now changes for x, so that even a constant
        density will look warped.

        :param ylim: y-axis limit
        :param scale: Scale of y-axis
        :param bins: Array of bin boundaries
        :param title: Title of the plot
        :param labels: Labels for the DFEs
        :param errors: Array of errors
        :param values: Array of values
        :param file: File path to save plot to
        :param show: Whether to show plot
        :param scale_density: Whether to scale the density by the bin size
        :param ax: Axes to plot on
        :return: Axes
        """
        from fastdfe.discretization import get_midpoints_and_spacing

        n_bins = len(bins) - 1
        n_dfes = len(values)

        # get interval sizes
        _, interval_sizes = get_midpoints_and_spacing(bins)

        for i in range(n_dfes):
            x = np.arange(n_bins)

            # plot DFE
            ax.plot(x,
                    values[i] if scale_density else values[i] / interval_sizes,
                    label=labels[i] if labels is not None else None)

            # plot error bars
            if errors is not None and errors[i] is not None:
                ax.fill_between(
                    x=x,
                    y1=(values[i] - errors[i][0]) if scale_density else (values[i] - errors[i][0]) / interval_sizes,
                    y2=(values[i] + errors[i][1]) if scale_density else (values[i] + errors[i][1]) / interval_sizes,
                    alpha=0.2
                )

        # customize x-ticks
        Visualization.adjust_ticks_show_s(bins)

        # use log scale if specified
        if scale == 'log':
            ax.set_yscale('log')
            ax.set_ylim(bottom=ylim)

        # show legend if labels were given
        if labels is not None:
            plt.legend(prop=dict(size=8))

        # remove x-margins
        ax.set_xmargin(0)

        ax.set_title(title)

        return ax

    @staticmethod
    def adjust_ticks_show_s(s: np.ndarray):
        """
        Adjust x-ticks to show bin values.

        :return: Array of selection coefficients
        """

        n_bins = len(s) - 1
        ax = plt.gca()

        ticks = ax.get_xticks()
        ticks_new = ["{:.0e}".format(s[int(l)]) if 0 <= int(l) < n_bins else None for l in ticks]

        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks_new)

    @staticmethod
    def plot_sfs_comparison(
            ax: plt.Axes,
            spectra: List[Spectrum] | np.ndarray,
            labels: List[str] = [],
            file: str = None,
            show: bool = True,
            title: str = 'SFS comparison',
            use_subplots: bool = False,
            show_monomorphic: bool = False,
    ) -> plt.Axes:
        """
        Plot SFS comparison.

        :param use_subplots: Whether to use subplots
        :param show_monomorphic: Whether to show monomorphic counts
        :param title: Title of the plot
        :param labels: Labels for the SFSs
        :param spectra: List of spectrum objects in the same order as the labels or a 2D array
        :param file: File path to save plot to
        :param show: Whether to show plot
        :param ax: Axes to plot on
        :return: Axes
        """
        # plot modelled vs observed non-neutral SFS
        ax = Visualization.plot_spectra(
            spectra=spectra,
            labels=labels,
            use_subplots=use_subplots,
            show_monomorphic=show_monomorphic,
            title=title,
            ax=ax
        )

        return ax

    @staticmethod
    @clear_show_save
    def plot_inferred_parameters(
            ax: plt.Axes,
            values: Dict[str, np.ndarray],
            labels: list | np.ndarray,
            param_names: list | np.ndarray,
            errors: Dict[str, np.ndarray | None],
            file: str = None,
            show: bool = True,
            title: str = 'parameter estimates',
            legend: bool = True,
            scale: Literal['lin', 'log', 'symlog'] = 'log'
    ) -> plt.Axes:
        """
        Visualize the inferred parameters and their confidence intervals.
        using a bar plot. Note that there problems with parameters that span 0.

        :param labels: Unique labels for the DFEs
        :param param_names: Labels for the parameters
        :param scale: Whether to use a linear or log scale
        :param title: Title of the plot
        :param legend: Whether to show the legend
        :param errors: Dictionary of errors with the parameter in the same order as ``labels``
        :param values: Dictionary of parameter values with the parameter in the same order as ``labels``
        :param file: File path to save plot to
        :param show: Whether to show plot
        :param ax: Axes to plot on
        :return: Axes
        """
        n_types = len(values)

        width_total = 0.9
        width = width_total / n_types

        # number of parameters
        n_params = len(param_names)

        # iterate over types
        for i, vals, errs in zip(range(n_types), values.values(), errors.values()):

            x = np.arange(n_params) + i * width

            def name_to_label(key: str) -> str:
                """
                Add parameter name to label.

                :param key: Parameter name
                :return: Label
                """
                # define new names for some parameters
                label_mapping = dict(
                    alpha='α',
                    eps='ε'
                )

                # map to new name
                label = label_mapping[key] if key in label_mapping else key

                # get index of parameter
                k = np.where(np.array(param_names) == key)[0][0]

                # check parameter value and add minus sign if negative
                if vals[k] >= 0:
                    return '$' + label + '$'

                return '$-' + key + '$'

            # append a minus sign to negative parameters values
            xlabels = np.array([name_to_label(key) for key in param_names])

            # flip errors for negative parameters
            if errs is not None:
                errs = np.array([err if vals[i] >= 0 else err[::-1] for i, err in enumerate(errs.T)]).T

            # Plot bars.
            # Note that we plot the absolute value of the parameter
            bars = ax.bar(
                x=x,
                height=np.abs(vals),
                yerr=errs,
                error_kw=dict(
                    capsize=width * 7
                ),
                label=labels[i],
                width=width,
                linewidth=0,
                hatch=Visualization.get_hatch(i, labels),
                color=Visualization.get_color(labels[i], labels) if labels is not None else None
            )

            Visualization.darken_edge_colors(bars)

            # customize x-ticks
            x = np.arange(n_params)
            ax.set_xticks([i + (width_total - width) / 2 for i in x], x)
            ax.set_xticklabels(xlabels)

        # show legend if specified
        if legend:
            plt.legend(prop=dict(size=8), loc='upper right')

        # set title
        ax.set_title(title)

        # change to log-scale if specified
        if scale == 'log':
            ax.set_yscale('symlog', linthresh=1e-3)

        # remove x-margins
        ax.autoscale(tight=True, axis='x')

        return ax

    @staticmethod
    def get_color(
            label: str,
            labels: List[str],
            get_group: Callable = lambda x: x.split('.')[-1]
    ) -> str:
        """
        Get color for specified label.

        :param label: Label to get color for
        :param labels: List of labels
        :param get_group: Function to get group from label
        :return: Color string
        """
        # determine unique groups
        groups = np.unique(np.array([get_group(l) for l in labels]))

        # determine group of current label
        group = get_group(label)

        # determine index of group
        i = np.where(groups == group)[0][0] if sum(groups == group) > 0 else 0

        # return color
        return f'C{i}'

    @staticmethod
    def get_hatch(i: int, labels: List[str] = None) -> str | None:
        """
        Get hatch style for specified index i.

        :param labels: List of labels
        :param i: Index
        :return: Hatch style
        """

        # determine whether hatch style should be used
        if labels is None or len(labels) < 1 or '.' not in labels[i]:
            return None

        # determine unique prefixes
        prefixes = set([label.split('.')[0] for label in labels if '.' in label])
        hatch_styles = ['/////', '\\\\\\\\\\', '***', 'ooo', 'xxx', '...']

        prefix = labels[i].split('.')[0]
        prefix_index = list(prefixes).index(prefix)

        return hatch_styles[prefix_index % len(hatch_styles)]

    @staticmethod
    def plot_spectra(
            spectra: List[Spectrum],
            ax: plt.Axes,
            labels: List[str] = [],
            log_scale: bool = False,
            use_subplots: bool = False,
            show_monomorphic: bool = False,
            title: str = None,
            n_ticks=10,
            file: str = None,
            show: bool = True
    ) -> plt.Axes:
        """
        Plot the given 1D spectra.

        :param show_monomorphic: Whether to show monomorphic site counts
        :param n_ticks: Number of x-ticks to use
        :param ax: Axes to plot on
        :param use_subplots: Whether to use subplots
        :param title: Title of plot
        :param spectra: List of spectra to plot
        :param labels: List of labels for each spectrum
        :param log_scale: Whether to use logarithmic y-scale
        :param file: File to save plot to
        :param show: Whether to show the plot
        :return: Axes
        """
        if ax is None:
            plt.clf()
            _, ax = plt.subplots()

        if use_subplots:
            n_plots = len(spectra)
            n_rows = int(np.ceil(np.sqrt(n_plots)))
            n_cols = int(np.ceil(np.sqrt(n_plots)))
            fig = plt.figure(figsize=(6.4 * n_cols ** (1 / 3), 4.8 * n_rows ** (1 / 3)))
            axes = fig.subplots(ncols=n_cols, nrows=n_rows, squeeze=False).flatten()

            # plot spectra individually
            for i in range(n_plots):
                Visualization.plot_spectra(
                    spectra=[spectra[i]],
                    labels=[labels[i]] if len(labels) else [],
                    ax=axes[i],
                    n_ticks=15 // min(2, n_cols),
                    log_scale=log_scale,
                    show_monomorphic=show_monomorphic
                )

            # make empty plots invisible
            [ax.set_visible(False) for ax in axes[n_plots:]]

            # show and save plot
            Visualization.show_and_save(file, show)

            return plt.gca()

        # determine sample size and width
        n = spectra[0].n
        width_total = 0.9
        width = width_total / len(spectra)

        x = np.arange(n + 1) if show_monomorphic else np.arange(1, n)

        # iterator over spectra and draw bars
        for i, sfs in enumerate(spectra):
            indices = x + i * width
            heights = sfs.to_list() if show_monomorphic else sfs.to_list()[1:-1]

            bars = ax.bar(
                x=indices,
                height=heights,
                width=width,
                label=labels[i] if len(labels) else None,
                linewidth=0,
                hatch=Visualization.get_hatch(i, labels)
            )

            Visualization.darken_edge_colors(bars)

        # adjust ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        indices_ticks = x

        # filter ticks
        if n > n_ticks:
            indices_ticks = indices_ticks[indices_ticks % max(int(np.ceil(n / n_ticks)), 1) == 1]

        ax.set_xticks([i + (width_total - width) / 2 for i in indices_ticks], indices_ticks)

        ax.set_xlabel('allele count')

        # remove x-margins
        ax.autoscale(tight=True, axis='x')

        if log_scale:
            ax.set_yscale('log')

        if len(labels) == 1:
            ax.set_title(labels[0])
        elif len(labels) > 1:

            # set title
            ax.set_title(title)

            ax.legend(prop=dict(size=8))

        # show and save plot
        Visualization.show_and_save(file, show)

        return ax

    @staticmethod
    def darken_edge_colors(bars: BarContainer):
        """
        Darken the edge color of the given bars.

        :param bars: Bars to darken
        """
        for bar in bars:
            color = bar.get_facecolor()
            edge_color = Visualization.darken_color(color, amount=0.75)
            bar.set_edgecolor(edge_color)

    @staticmethod
    def darken_color(color, amount=0.5) -> tuple:
        """
        Darken a color.

        :param color: Color to darken
        :param amount: Amount to darken
        :return: Darkened color as tuple
        """
        c = mcolors.to_rgba(color)

        return c[0] * amount, c[1] * amount, c[2] * amount, c[3]

    @staticmethod
    @clear_show_save
    def plot_pdf(
            model: Parametrization,
            params: dict,
            s: np.array,
            ax: plt.Axes,
            file: str = None,
            show: bool = True
    ) -> plt.Axes:
        """
        Plot PDF of given parametrization.

        :param model: DFE parametrization
        :param params: Parameters to be used for parametrization
        :param s: Selection coefficients
        :param file: File to save plot to
        :param show: Whether to show plot
        :param ax: Axes to plot on
        :return: Axes
        """
        ax.plot(np.arange(len(s)), model.get_pdf(**params)(s))

        # customize x-ticks
        Visualization.adjust_ticks_show_s(s)

        # remove x-margins
        ax.margins(x=0)

        return ax

    @staticmethod
    @clear_show_save
    def plot_cdf(
            model: Parametrization,
            params: dict,
            s: np.array,
            ax: plt.Axes,
            file: str = None,
            show: bool = True
    ) -> plt.Axes:
        """
        Plot CDF of given parametrization.

        :param model: DFE parametrization
        :param params: Parameters to be used for parametrization
        :param s: Selection coefficients
        :param ax: Axes to plot on
        :param file: File to save plot to
        :param show: Whether to show plot
        :return: Axes
        """
        ax.plot(np.arange(len(s)), model.get_cdf(**params)(s))

        # customize x-ticks
        Visualization.adjust_ticks_show_s(s)

        # remove x-margins
        ax.margins(x=0)

        return ax

    @staticmethod
    @clear_show_save
    def plot_likelihoods(
            likelihoods: list | np.ndarray,
            file: str, show: bool,
            title: str,
            ax: plt.Axes,
            scale: Literal['lin', 'log', 'symlog'] = 'lin'
    ) -> plt.Axes:
        """
        A scatter plot of the likelihoods specified.

        :param scale: Scale of y-axis
        :param likelihoods: Likelihoods to plot
        :param file: File to save plot to
        :param show: Whether to show plot
        :param title: Title of plot
        :param ax: Axes to plot on
        :return: Axes
        """
        # plot
        ax.scatter(np.arange(len(likelihoods)), likelihoods)

        # set title
        ax.set_title(title)

        # use integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # set labels
        ax.set_xlabel('iteration')
        ax.set_ylabel('lnl')

        if scale == 'log':
            ax.set_yscale('symlog')

        return ax

    @staticmethod
    @clear_show_save
    def plot_buckets_sizes(
            n_intervals: int,
            bins: list | np.ndarray,
            sizes: list | np.ndarray,
            title: str,
            file: str,
            show: bool,
            ax: plt.Axes
    ) -> plt.Axes:
        """
        A line plot of the bucket sizes.

        :param bins: Bins of the histogram
        :param n_intervals: Number of intervals
        :param sizes: Sizes of the buckets
        :param title: Title of plot
        :param file: File to save plot to
        :param show: Whether to show plot
        :param ax: Axes to plot on
        :return: Axes
        """
        # plot line
        ax.plot(np.arange(n_intervals), sizes)

        # use log scale
        ax.set_yscale('log')
        ax.set_title(title)

        # customize x-ticks
        Visualization.adjust_ticks_show_s(bins)

        # remove x-margins
        ax.set_xmargin(0)

        return ax

    @staticmethod
    @clear_show_save
    def plot_nested_likelihoods(
            P: np.ndarray,
            labels_x: list,
            labels_y: list,
            ax: plt.Axes,
            file: str = None,
            show: bool = True,
            cmap: str = None,
            title: str = None
    ) -> plt.Axes:
        """
        Plot p-values of nested likelihoods.

        :param P: Matrix of p-values
        :param labels_x: Labels for x-axis
        :param labels_y: Labels for y-axis
        :param file: File to save plot to
        :param show: Whether to show plot
        :param cmap: Colormap to use
        :param title: Title of plot
        :param ax: Axes to plot on
        :return: Axes
        """

        def format_number(x: float | int | None) -> float | int | str:
            """
            Format number to be displayed.
            """
            if x == 0 or x is None:
                return 0

            if x < 0.0001:
                return "{:.1e}".format(x)

            return np.round(x, 4)

        # determine values to display
        annot = np.vectorize(lambda x: str(format_number(x)))(P)
        annot[np.equal(P, None)] = '-'

        # change to 1 to get a nicer color
        P[np.equal(P, None)] = 1

        # make the cbar have the same height as the heatmap
        cbar_ax = make_axes_locatable(ax).new_horizontal(size="4%", pad=0.15)
        plt.gcf().add_axes(cbar_ax)

        # default color map
        if cmap is None:
            cmap = colors.LinearSegmentedColormap.from_list('_', plt.get_cmap('inferno')(np.linspace(0.3, 1, 100)))

        # plot heatmap
        sns.heatmap(
            P.astype(float),
            ax=ax,
            cbar_ax=cbar_ax,
            cmap=cmap,
            norm=LogNorm(
                vmin=1e-10,
                vmax=1
            ),
            annot=annot,
            fmt="",
            square=True,
            linewidths=0.5,
            linecolor='#cccccc',
            cbar_kws=dict(
                label='p-value'
            )
        )

        # adjust tick labels
        ax.set_xticklabels([l.replace('_', ' ') for l in labels_x], rotation=45)
        ax.set_yticklabels([l.replace('_', ' ') for l in labels_y], rotation=0)

        # set title
        ax.set_title(title)

        return ax

    @staticmethod
    def plot_covariates(
            covariates: Dict[str, 'Covariate'],
            params_marginal: Dict[str, Dict[str, float]],
            params_joint: Dict[str, Dict[str, float]],
            axs: List[plt.Axes],
            scale: Literal['lin', 'log', 'symlog'] = 'log',
            file: str = None,
            show: bool = True,
            **kwargs
    ) -> List[plt.Axes]:
        """
        Plot covariates.

        :param covariates: Covariates to plot
        :param params_marginal: Marginal parameters indexed by type
        :param params_joint: Unpacked joint parameters indexed by type
        :param axs: List of axes, one for each covariate
        :param scale: Scale of y-axis
        :param file: File to save plot to
        :param show: Whether to show plot
        :return: Axes
        """
        if len(covariates) == 0:
            logger.info("There are no covariates to be plotted.")

            return axs

        types = list(params_joint.keys())

        num_covariates = len(covariates)

        n_cols = min(3, num_covariates)
        n_rows = int(np.ceil(num_covariates / n_cols))

        # create axes if not provided
        if not isinstance(axs, np.ndarray | list):
            plt.clf()
            _, axs = plt.subplots(n_rows, n_cols, figsize=(6.4 * n_cols, 4.8 * n_rows), squeeze=False)

        n_intervals = len(types)
        width_total = 0.9
        width = width_total / 2

        for i, (p, cov) in enumerate(covariates.items()):
            ax = axs[i // n_cols, i % n_cols]

            x = np.arange(n_intervals)

            y1 = np.abs([params_marginal[t][cov.param] for t in types])
            y2 = np.abs([params_joint[t][cov.param] for t in types])

            # plot bars for marginal inference
            ax.bar(
                x=x,
                height=y1,
                width=width,
                label='marginal'
            )

            # plot bars for joint inference
            ax.bar(
                x=x + width,
                height=y2,
                width=width,
                label='joint'
            )

            # proper representation of parameter
            param_repr = f'${cov.param}$' if params_marginal[types[0]][cov.param] >= 0 else f'$-{cov.param}$'

            # set title
            ax.set_title(f"param={param_repr}, {p}={params_joint[types[0]][p]}")

            # adjust ticks
            ax.set_xticks(np.arange(len(types)) + width / 2)
            ax.set_xticklabels(types)

            # set legend
            ax.legend()

            # set y-scale
            if scale == 'log':
                ax.set_yscale('log')

        # show and save plot
        Visualization.show_and_save(file, show)

        return axs

    @clear_show_save
    def plot_interval_density(
            self,
            ax: plt.Axes,
            density: np.ndarray,
            intervals: list | np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            interval_labels: List[str] = None,
            file: str = None,
            show: bool = True,
            color: str = 'C0',

    ) -> plt.Axes:
        """
        Plot density of the discretization intervals chosen.

        :param density:
        :param color: Color of the bars.
        :param file: File to save plot to.
        :param show: Whether to show plot.
        :param interval_labels: Labels for the intervals.
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :param ax: Axes to plot on.
        :return: Axes object
        """
        # number of intervals
        n_intervals = len(intervals)

        # x-values
        x = np.arange(n_intervals - 1)

        # plot plot
        sns.barplot(x=x, y=density / density.sum(), color=color, ax=ax)

        # set labels
        ax.set(xlabel='S', ylabel='fraction')

        # determine x-labels
        if interval_labels is None:
            xlabels = [Visualization.interval_to_string(intervals[i - 1], intervals[i]) for i in range(1, n_intervals)]
        else:
            xlabels = interval_labels

        # adjust x-ticks
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)

        # set title
        ax.set_title('interval density')

        # remove x-margins
        ax.autoscale(tight=True, axis='x')

        return ax
