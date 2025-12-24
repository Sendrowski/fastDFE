"""
Abstract inference class and static utilities.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-12"

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Tuple, Dict, Sequence

import jsonpickle
import numpy as np
import pandas as pd
from typing_extensions import Self

from .bootstrap import Bootstrap
from .parametrization import Parametrization, _from_string
from .utils import Serializable

logger = logging.getLogger("fastdfe")

class Inference:
    """
    Static utility methods for inference objects.
    """

    @staticmethod
    def plot_discretized(
            inferences: List['AbstractInference'],
            intervals: Sequence = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            point_estimate: Literal['original', 'mean', 'median'] = 'mean',
            file: str = None,
            show: bool = True,
            title: str = 'discretized DFEs',
            labels: Sequence = None,
            ax: 'plt.Axes' = None,
            kwargs_legend: dict = dict(prop=dict(size=8)),
            **kwargs

    ) -> 'plt.Axes':
        """
        Visualize several discretized DFEs given by the list of inference objects.

        :param inferences: List of inference objects.
        :param intervals: Intervals over ``(-inf, inf)`` to use for discretization.
        :param confidence_intervals: Whether to plot confidence intervals.
        :param ci_level: Confidence level for confidence intervals.
        :param bootstrap_type: Type of bootstrap to use for confidence intervals.
        :param point_estimate: Whether to use 'original' MLE values, 'mean' or 'median' of bootstraps as point estimate.
        :param file: Path to file to save the plot to.
        :param show: Whether to show the plot.
        :param title: Title of the plot.
        :param labels: Labels for the DFEs.
        :param kwargs: Additional arguments for the plot.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :return: Axes of the plot.
        """
        from .visualization import Visualization

        # get data from inference objects
        values = []
        errors = []
        for i, inference in enumerate(inferences):
            val, errs = inference.get_discretized(
                intervals=np.array(intervals),
                confidence_intervals=confidence_intervals,
                ci_level=ci_level,
                bootstrap_type=bootstrap_type,
                point_estimate=point_estimate
            )

            values.append(val)
            errors.append(errs)

        # plot DFEs
        return Visualization.plot_discretized(
            values=values,
            errors=errors,
            labels=labels,
            file=file,
            show=show,
            intervals=np.array(intervals),
            title=title,
            ax=ax,
            kwargs_legend=kwargs_legend
        )

    @staticmethod
    def plot_continuous(
            inferences: List['AbstractInference'],
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            file: str = None,
            show: bool = True,
            title: str = 'continuous DFEs',
            labels: Sequence = None,
            scale: Literal['lin', 'log', 'symlog'] = 'lin',
            scale_density: bool = False,
            ax: 'plt.Axes' = None,
            kwargs_legend: dict = dict(prop=dict(size=8)),
            **kwargs

    ) -> 'plt.Axes':
        """
        Visualize several DFEs given by the list of inference objects.
        By default, the PDF is plotted as is. Due to the logarithmic scale on
        the x-axis, we may get a wrong intuition on how the mass is distributed,
        however. To get a better intuition, we can optionally scale the density
        by the x-axis interval size using ``scale_density = True``. This has the
        disadvantage that the density now changes for x, so that even a constant
        density will look warped.

        :param inferences: List of inference objects.
        :param intervals: Intervals to use for discretization.
        :param confidence_intervals: Whether to plot confidence intervals.
        :param ci_level: Confidence level for confidence intervals.
        :param bootstrap_type: Type of bootstrap to use for confidence intervals.
        :param file: Path to file to save the plot to.
        :param show: Whether to show the plot.
        :param title: Title of the plot.
        :param labels: Labels for the DFEs.
        :param scale: y-scale of the plot.
        :param scale_density: Whether to scale the density by the x-axis interval size.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :param kwargs: Additional arguments for the plot.
        :return: Axes of the plot.
        """
        from .visualization import Visualization

        # get data from inference objects
        values = []
        errors = []
        for i, inf in enumerate(inferences):
            val, errs = inf.get_discretized(
                intervals=intervals,
                confidence_intervals=confidence_intervals,
                ci_level=ci_level,
                bootstrap_type=bootstrap_type
            )

            values.append(val)
            errors.append(errs)

        # plot DFEs
        return Visualization.plot_continuous(
            bins=intervals,
            **locals()
        )

    @staticmethod
    def plot_inferred_parameters(
            inferences: List['AbstractInference'],
            labels: Sequence,
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            point_estimate: Literal['original', 'mean', 'median'] = 'mean',
            file: str = None,
            show: bool = True,
            title: str = 'parameter estimates',
            scale: Literal['lin', 'log', 'symlog'] = 'log',
            ax: 'plt.Axes' = None,
            kwargs_legend: dict = dict(prop=dict(size=8), loc='upper right'),
            **kwargs

    ) -> 'plt.Axes':
        """
        Visualize several discretized DFEs given by the list of inference objects.
        Note that the DFE parametrization needs to be the same for all inference objects.

        :param inferences: List of inference objects.
        :param labels: Unique labels for the DFEs.
        :param scale: y-scale of the plot.
        :param confidence_intervals: Whether to plot confidence intervals.
        :param ci_level: Confidence level for confidence intervals.
        :param bootstrap_type: Type of bootstrap to use for confidence intervals.
        :param point_estimate: Whether to use 'original' MLE values, 'mean' or 'median' of bootstraps as point estimate.
        :param file: Path to file to save the plot to.
        :param show: Whether to show the plot.
        :param title: Title of the plot.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :return: Axes of the plot.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :param kwargs: Additional arguments which are ignored.
        :raises ValueError: If no inference objects are given.
        """
        from .visualization import Visualization

        if len(inferences) == 0:
            raise ValueError('No inference objects given.')

        # get sorted list of parameter names
        param_names = sorted(list(inferences[0].get_bootstrap_param_names()))

        # get errors and values
        errors, values = Inference.get_errors_params_mle(
            bootstrap_type=bootstrap_type,
            ci_level=ci_level,
            confidence_intervals=confidence_intervals,
            inferences=inferences,
            labels=labels,
            param_names=param_names,
            point_estimate=point_estimate
        )

        return Visualization.plot_inferred_parameters(
            values=values,
            errors=errors,
            param_names=param_names,
            file=file,
            show=show,
            title=title,
            labels=labels,
            scale=scale,
            legend=len(labels) > 1,
            kwargs_legend=kwargs_legend,
            ax=ax,
        )

    @staticmethod
    def plot_inferred_parameters_boxplot(
            inferences: List['AbstractInference'],
            labels: Sequence,
            file: str = None,
            show: bool = True,
            title: str = 'parameter estimates',
            **kwargs
    ) -> 'plt.Axes':
        """
        Visualize several discretized DFEs given by the list of inference objects.
        Note that the DFE parametrization needs to be the same for all inference objects.

        :param inferences: List of inference objects.
        :param labels: Unique labels for the DFEs.
        :param file: Path to file to save the plot to.
        :param show: Whether to show the plot.
        :param title: Title of the plot.
        :param kwargs: Additional arguments for the plot.
        :return: Axes of the plot.
        :raises ValueError: If no inference objects are given or no bootstraps are found.
        """
        from .visualization import Visualization

        if len(inferences) == 0:
            raise ValueError('No inference objects given.')

        # get sorted list of parameter names
        param_names = sorted(list(inferences[0].get_bootstrap_param_names()))

        if inferences[0].bootstraps is None:
            raise ValueError('No bootstraps found.')

        # create dict of dataframes
        values = dict((k, inf.bootstraps) for k, inf in zip(labels, inferences))

        return Visualization.plot_inferred_parameters_boxplot(
            values=values,
            param_names=param_names,
            file=file,
            show=show,
            title=title,
        )

    @staticmethod
    def get_errors_params_mle(
            ci_level: float,
            confidence_intervals: bool,
            inferences: List['AbstractInference'],
            labels: Sequence,
            param_names: Sequence,
            bootstrap_type: Literal['percentile', 'bca'],
            point_estimate: Literal['original', 'mean', 'median'] = 'mean',
    ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray] | None], Dict[str, np.ndarray]]:
        """
        Get errors and values for MLE params of inferences.

        :param ci_level: Confidence level for confidence intervals.
        :param confidence_intervals: Whether to compute confidence intervals.
        :param inferences: List of inference objects.
        :param labels: Labels for the inferences.
        :param param_names: Names of the parameters to get errors and values for.
        :param bootstrap_type: Type of bootstrap to use.
        :param point_estimate: Whether to use 'original' MLE values, 'mean' or 'median' of bootstraps as point estimate.
        :return: dictionary of errors and dictionary of center values indexed by labels.
        """
        errors, values, center = {}, {}, {}
        for label, inf in zip(labels, inferences):

            values[label] = list(inf.get_bootstrap_params()[k] for k in param_names)

            # whether to compute errors
            if confidence_intervals and inf.bootstraps is not None:

                # compute errors
                center[label], errors[label], _ = Bootstrap.get_errors(
                    values=values[label],
                    bs=inf.bootstraps[param_names].to_numpy(),
                    bootstrap_type=bootstrap_type,
                    ci_level=ci_level,
                    point_estimate=point_estimate,
                )
            else:
                center[label] = np.array(values[label])
                errors[label] = None

        return errors, center

    @staticmethod
    def get_discretized(
            inferences: List['AbstractInference'],
            labels: Sequence,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            point_estimate: Literal['original', 'mean', 'median'] = 'mean'
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Optional[np.ndarray]]]:
        """
        Get values and errors of discretized DFE.

        :param inferences: List of inference objects.
        :param labels: Labels for the DFEs.
        :param bootstrap_type: Type of bootstrap to use
        :param ci_level: Confidence interval level
        :param confidence_intervals: Whether to compute confidence intervals
        :param intervals: Array of interval boundaries over ``(-inf, inf)`` yielding ``intervals.shape[0] - 1`` bars.
        :param point_estimate: Whether to use 'original' MLE values, 'mean' or 'median' of bootstraps as point estimate.
        :return: Dictionary of values and dictionary of errors indexed by labels.
        """
        values = {}
        errors = {}

        for label, inf in zip(labels, inferences):

            if confidence_intervals and inf.bootstraps is not None:
                # get bootstraps and errors if specified
                values[label], errors[label], _ = Inference.get_stats_discretized(
                    params=inf.get_bootstrap_params(),
                    bootstraps=inf.bootstraps,
                    model=inf.model,
                    ci_level=ci_level,
                    intervals=intervals,
                    bootstrap_type=bootstrap_type,
                    point_estimate=point_estimate
                )
            else:
                # otherwise just get discretized values
                values[label] = Inference.compute_histogram(
                    params=inf.get_bootstrap_params(),
                    model=inf.model,
                    intervals=intervals
                )
                errors[label] = None

        return values, errors

    @staticmethod
    def get_stats_discretized(
            params: dict,
            bootstraps: pd.DataFrame,
            model: Parametrization | str,
            ci_level: float = 0.05,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            point_estimate: Literal['original', 'mean', 'median'] = 'mean'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute errors and confidence interval for a discretized DFE.

        :param params: Parameters of the model
        :param bootstraps: Bootstrapped samples
        :param model: DFE parametrization
        :param ci_level: Confidence interval level
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :param bootstrap_type: Type of bootstrap
        :param point_estimate: Whether to use 'original' MLE values, 'mean' or 'median' of bootstraps as point estimate.
        :return: Center values, errors around center, and confidence intervals.
        """
        # discretize MLE DFE
        values = Inference.compute_histogram(model, params, intervals)

        # calculate bootstrapped histograms
        # get discretized DFE per bootstrap sample
        bs = np.array([Inference.compute_histogram(model, dict(r), intervals) for _, r in bootstraps.iterrows()])

        return Bootstrap.get_errors(
            values=values,
            bs=bs,
            bootstrap_type=bootstrap_type,
            ci_level=ci_level,
            point_estimate=point_estimate
        )

    @staticmethod
    def compute_histogram(
            model: Parametrization | str,
            params: dict,
            intervals: np.ndarray
    ) -> np.ndarray:
        """
        Discretize the DFE given a DFE parametrization and its parameter values.

        :param model: DFE parametrization
        :param params: Parameters of the model
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :return: Discretized DFE
        """
        # discrete DFE
        y = _from_string(model)._discretize(params, intervals)

        # return normalized histogram
        return y / y.sum()


class AbstractInference(Serializable, ABC):
    """
    Base class for main Inference and polyDFE wrapper.
    """

    def __init__(self, **kwargs):
        """
        Initialize the inference.

        :param kwargs: Keyword arguments
        """
        self._logger = logger.getChild(self.__class__.__name__)

        self.bootstraps: Optional[pd.DataFrame] = None
        self.params_mle: Optional[dict] = None
        self.model: Optional[Parametrization] = None

    @abstractmethod
    def get_bootstrap_params(self) -> Dict[str, float]:
        """
        Get the parameters to be included in the bootstraps.

        :return: Parameters to be included in the bootstraps
        """
        pass

    def get_discretized(
            self,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            point_estimate: Literal['original', 'mean', 'median'] = 'mean'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get discretized DFE.

        :param intervals: Array of interval boundaries over ``(-inf, inf)`` yielding ``intervals.shape[0] - 1`` bins.
        :param confidence_intervals: Whether to return confidence intervals
        :param ci_level: Confidence interval level
        :param bootstrap_type: Type of bootstrap
        :param point_estimate: Whether to use 'original' MLE values, 'mean' or 'median' of bootstraps as point estimate.
        :return: Array of values and array of deviations
        """
        values, errors = Inference.get_discretized(
            inferences=[self],
            labels=['all'],
            intervals=intervals,
            confidence_intervals=confidence_intervals,
            ci_level=ci_level,
            bootstrap_type=bootstrap_type,
            point_estimate=point_estimate
        )

        return values['all'], errors['all']

    def plot_discretized(
            self,
            file: str = None,
            show: bool = True,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            point_estimate: Literal['original', 'mean', 'median'] = 'mean',
            title: str = 'discretized DFE',
            ax: 'plt.Axes' = None,
            kwargs_legend: dict = dict(prop=dict(size=8)),
    ) -> 'plt.Axes':
        """
        Plot discretized DFE.

        :param file: File to save the plot to
        :param show: Whether to show the plot
        :param intervals: Array of interval boundaries over ``(-inf, inf)`` yielding ``intervals.shape[0] - 1`` bars.
        :param confidence_intervals: Whether to plot confidence intervals
        :param ci_level: Confidence interval level
        :param bootstrap_type: Type of bootstrap
        :param point_estimate: Whether to use 'original' MLE values, 'mean' or 'median' of bootstraps as point estimate.
        :param title: Title of the plot
        :param ax: Axes to plot on. Only for Python visualization backend.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :return: Axes
        """
        return Inference.plot_discretized(
            inferences=[self],
            file=file,
            show=show,
            intervals=intervals,
            confidence_intervals=confidence_intervals,
            ci_level=ci_level,
            bootstrap_type=bootstrap_type,
            point_estimate=point_estimate,
            title=title,
            kwargs_legend=kwargs_legend,
            ax=ax
        )

    @abstractmethod
    def get_bootstrap_param_names(self) -> List[str]:
        """
        Get the names of the parameters to be included in the bootstraps.
        """
        pass
