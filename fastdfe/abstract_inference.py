"""
Abstract inference class and static utilities.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-12"

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Literal, Tuple, Dict

import jsonpickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing_extensions import Self

from .bootstrap import Bootstrap
from .parametrization import Parametrization, _from_string
from .visualization import Visualization

logger = logging.getLogger("fastdfe")


class Inference:
    """
    Static utility methods for inference objects.
    """

    @staticmethod
    def plot_discretized(
            inferences: List['AbstractInference'],
            intervals: list | np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            file: str = None,
            show: bool = True,
            title: str = 'discretized DFEs',
            labels: list | np.ndarray = None,
            ax: plt.Axes = None,
            kwargs_legend: dict = dict(prop=dict(size=8)),
            **kwargs

    ) -> plt.Axes:
        """
        Visualize several discretized DFEs given by the list of inference objects.

        :param inferences: List of inference objects.
        :param intervals: Intervals to use for discretization.
        :param confidence_intervals: Whether to plot confidence intervals.
        :param ci_level: Confidence level for confidence intervals.
        :param bootstrap_type: Type of bootstrap to use for confidence intervals.
        :param file: Path to file to save the plot to.
        :param show: Whether to show the plot.
        :param title: Title of the plot.
        :param labels: Labels for the DFEs.
        :param kwargs: Additional arguments for the plot.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :return: Axes of the plot.
        """
        # get data from inference objects
        values = []
        errors = []
        for i, inference in enumerate(inferences):
            val, errs = inference.get_discretized(
                intervals=np.array(intervals),
                confidence_intervals=confidence_intervals,
                ci_level=ci_level,
                bootstrap_type=bootstrap_type
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
            labels: list | np.ndarray = None,
            scale: Literal['lin', 'log', 'symlog'] = 'lin',
            scale_density: bool = False,
            ax: plt.Axes = None,
            kwargs_legend: dict = dict(prop=dict(size=8)),
            **kwargs

    ) -> plt.Axes:
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
            labels: list | np.ndarray,
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            file: str = None,
            show: bool = True,
            title: str = 'parameter estimates',
            scale: Literal['lin', 'log', 'symlog'] = 'log',
            ax: plt.Axes = None,
            kwargs_legend: dict = dict(prop=dict(size=8), loc='upper right'),
            **kwargs

    ) -> plt.Axes:
        """
        Visualize several discretized DFEs given by the list of inference objects.
        Note that the DFE parametrization needs to be the same for all inference objects.

        :param inferences: List of inference objects.
        :param labels: Unique labels for the DFEs.
        :param scale: y-scale of the plot.
        :param legend: Whether to show a legend.
        :param confidence_intervals: Whether to plot confidence intervals.
        :param ci_level: Confidence level for confidence intervals.
        :param bootstrap_type: Type of bootstrap to use for confidence intervals.
        :param file: Path to file to save the plot to.
        :param show: Whether to show the plot.
        :param title: Title of the plot.
        :param ax: Axes to plot on. Only for Python visualization backend.
        :return: Axes of the plot.
        :param kwargs_legend: Keyword arguments passed to :meth:`plt.legend`. Only for Python visualization backend.
        :param kwargs: Additional arguments which are ignored.
        :raises ValueError: If no inference objects are given.
        """
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
            param_names=param_names
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
            labels: list | np.ndarray,
            file: str = None,
            show: bool = True,
            title: str = 'parameter estimates',
            **kwargs
    ) -> plt.Axes:
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
    def get_cis_params_mle(
            bootstrap_type: Literal['percentile', 'bca'],
            ci_level: float,
            inferences: List['AbstractInference'],
            labels: list | np.ndarray,
            param_names: list | np.ndarray
    ) -> dict[str, Optional[Dict[str, Tuple[float, float]]]]:
        """
        Get confidence intervals for the MLE parameters.

        :param bootstrap_type: Type of bootstrap to use for confidence intervals.
        :param ci_level: Confidence level for confidence intervals.
        :param inferences: List of inference objects.
        :param labels: Labels for the DFEs.
        :param param_names: Names of the parameters.
        :return: Dictionary of confidence intervals indexed by labels.
        """
        cis = {}
        for label, inf in zip(labels, inferences):

            if inf.bootstraps is not None:

                values = list(inf.get_bootstrap_params()[k] for k in param_names)

                res = Bootstrap.get_errors(
                    values=values,
                    bs=inf.bootstraps[param_names].to_numpy(),
                    bootstrap_type=bootstrap_type,
                    ci_level=ci_level
                )

                cis[label] = {k: tuple(res[1][:, i]) for i, k in enumerate(param_names)}
            else:
                cis[label] = None

        return cis

    @staticmethod
    def get_errors_params_mle(
            bootstrap_type: Literal['percentile', 'bca'],
            ci_level: float,
            confidence_intervals: bool,
            inferences: List['AbstractInference'],
            labels: list | np.ndarray,
            param_names: list | np.ndarray
    ) -> (Dict[str, Tuple[np.ndarray, np.ndarray] | None], Dict[str, np.ndarray]):
        """
        Get errors and values for MLE params of inferences.

        :param bootstrap_type: Type of bootstrap to use.
        :param ci_level: Confidence level for confidence intervals.
        :param confidence_intervals: Whether to compute confidence intervals.
        :param inferences: List of inference objects.
        :param labels: Labels for the inferences.
        :param param_names: Names of the parameters to get errors and values for.
        :return: dictionary of errors and dictionary of values
        """
        errors = {}
        values = {}
        for label, inf in zip(labels, inferences):

            values[label] = list(inf.get_bootstrap_params()[k] for k in param_names)

            # whether to compute errors
            if confidence_intervals and inf.bootstraps is not None:

                # use mean of bootstraps instead of original values
                if bootstrap_type == 'percentile':
                    values[label] = inf.bootstraps[param_names].mean().to_list()

                # compute errors
                errors[label], _ = Bootstrap.get_errors(
                    values=values[label],
                    bs=inf.bootstraps[param_names].to_numpy(),
                    bootstrap_type=bootstrap_type,
                    ci_level=ci_level
                )
            else:
                errors[label] = None

        return errors, values

    @staticmethod
    def get_discretized(
            inferences: List['AbstractInference'],
            labels: list | np.ndarray,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile'

    ) -> (np.ndarray, Optional[np.ndarray]):
        """
        Get values and errors of discretized DFE.

        :param inferences: List of inference objects.
        :param labels: Labels for the DFEs.
        :param bootstrap_type: Type of bootstrap to use
        :param ci_level: Confidence interval level
        :param confidence_intervals: Whether to compute confidence intervals
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bars.
        :return: Array of values and array of errors
        """
        values = {}
        errors = {}

        for label, inf in zip(labels, inferences):

            if confidence_intervals and inf.bootstraps is not None:
                # get bootstraps and errors if specified
                errs, _, bs, means, vals = Inference.get_stats_discretized(
                    params=inf.get_bootstrap_params(),
                    bootstraps=inf.bootstraps,
                    model=inf.model,
                    ci_level=ci_level,
                    intervals=intervals,
                    bootstrap_type=bootstrap_type
                )
            else:
                # otherwise just get discretized values
                vals = Inference.compute_histogram(
                    params=inf.get_bootstrap_params(),
                    model=inf.model,
                    intervals=intervals
                )
                errs, means, bs = None, None, None

            # whether to use the mean of all bootstraps instead of the original values
            use_means = confidence_intervals and inf.bootstraps is not None and bootstrap_type == 'percentile'

            if use_means:
                vals = np.mean(bs, axis=0)

            values[label] = vals
            errors[label] = errs

        return values, errors

    @staticmethod
    def get_stats_discretized(
            params: dict,
            bootstraps: pd.DataFrame,
            model: Parametrization | str,
            ci_level: float = 0.05,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile'

    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Compute errors and confidence interval for a discretized DFE.

        :param params: Parameters of the model
        :param bootstraps: Bootstrapped samples
        :param model: DFE parametrization
        :param ci_level: Confidence interval level
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :param bootstrap_type: Type of bootstrap
        :return: Arrays of errors, confidence intervals, bootstraps, means and values
        """
        # discretize MLE DFE
        values = Inference.compute_histogram(model, params, intervals)

        # calculate bootstrapped histograms
        # get discretized DFE per bootstrap sample
        bs = np.array([Inference.compute_histogram(model, dict(r), intervals) for _, r in bootstraps.iterrows()])

        errors, cis = Bootstrap.get_errors(
            values=values,
            bs=bs,
            bootstrap_type=bootstrap_type,
            ci_level=ci_level
        )

        # calculate mean values
        means = np.mean(bs, axis=0)

        return errors, cis, bs, means, values

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


class AbstractInference(ABC):
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
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile'
    ) -> (np.ndarray, np.ndarray):
        """
        Get discretized DFE.

        :param bootstrap_type: Type of bootstrap
        :param ci_level: Confidence interval level
        :param confidence_intervals: Whether to return confidence intervals
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :return: Array of values and array of errors
        """
        values, errors = Inference.get_discretized(
            inferences=[self],
            labels=['all'],
            intervals=intervals,
            confidence_intervals=confidence_intervals,
            ci_level=ci_level,
            bootstrap_type=bootstrap_type
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
            title: str = 'discretized DFE',
            ax: plt.Axes = None,
            kwargs_legend: dict = dict(prop=dict(size=8)),
    ) -> plt.Axes:
        """
        Plot discretized DFE.

        :param title: Title of the plot
        :param bootstrap_type: Type of bootstrap
        :param ci_level: Confidence interval level
        :param confidence_intervals: Whether to plot confidence intervals
        :param file: File to save the plot to
        :param show: Whether to show the plot
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bars.
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
            title=title,
            kwargs_legend=kwargs_legend,
            ax=ax
        )

    def get_cis_params_mle(
            self,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            ci_level: float = 0.05,
            param_names: Optional[list[str]] = None
    ):
        """
        Get confidence intervals for the parameters.

        :param bootstrap_type: Type of bootstrap
        :param ci_level: Confidence interval level
        :param param_names: Names of the parameters to return confidence intervals for
        :return: Confidence intervals for the parameters
        """
        if param_names is None:
            param_names = self.get_bootstrap_param_names()

        return Inference.get_cis_params_mle(
            inferences=[self],
            bootstrap_type=bootstrap_type,
            ci_level=ci_level,
            param_names=param_names,
            labels=['all']
        )['all']

    def to_json(self) -> str:
        """
        Serialize object.

        :return: JSON string
        """
        return jsonpickle.encode(self, indent=4, warn=True)

    def to_file(self, file: str):
        """
        Save object to file.

        :param file: File to save to
        """
        with open(file, 'w') as fh:
            fh.write(self.to_json())

    @classmethod
    def from_json(cls, json: str, classes=None) -> Self:
        """
        Unserialize object.

        :param classes: Classes to be used for unserialization
        :param json: JSON string
        """
        return jsonpickle.decode(json, classes=classes)

    @classmethod
    def from_file(cls, file: str, classes=None) -> Self:
        """
        Load object from file.

        :param classes: Classes to be used for unserialization
        :param file: File to load from
        """
        with open(file, 'r') as fh:
            return cls.from_json(fh.read(), classes)

    @abstractmethod
    def get_bootstrap_param_names(self) -> List[str]:
        """
        Get the names of the parameters to be included in the bootstraps.
        """
        pass
