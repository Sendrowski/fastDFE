"""
Abstract inference class and static utilities.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-12"

from abc import ABC, abstractmethod
from typing import List, Optional, Literal
from typing_extensions import Self

import jsonpickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .bootstrap import Bootstrap
from .parametrization import Parametrization, from_string
from .visualization import Visualization


class Inference:
    """
    Static utility methods for inference objects.
    """

    @staticmethod
    def plot_discretized(
            inferences: List['AbstractInference'],
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            file: str = None,
            show: bool = True,
            title: str = 'discretized DFEs',
            labels: list | np.ndarray = None,
            ax: plt.Axes = None,
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
        :param ax: Axes of the plot.
        :return: Axes of the plot.
        """
        # get data from inference objects
        values = []
        errors = []
        for i, inference in enumerate(inferences):
            val, errs = inference.get_discretized(
                intervals=intervals,
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
            intervals=intervals,
            title=title,
            ax=ax
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
        :param ax: Axes of the plot.
        :param kwargs: Additional arguments for the plot.
        :return: Axes of the plot.
        """
        # get data from inference objects
        values = []
        errors = []
        for i, inference in enumerate(inferences):
            val, errs = inference.get_discretized(
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
            legend: bool = True,
            ax: plt.Axes = None,
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
        :param ax: Axes of the plot.
        :param kwargs: Additional arguments for the plot.
        :return: Axes of the plot.
        :raises ValueError: If no inference objects are given.
        """
        if len(inferences) == 0:
            raise ValueError('No inference objects given.')

        # get sorted list of parameter names
        param_names = sorted(list(inferences[0].get_bootstrap_param_names()))

        # get errors and values
        errors, values = Inference.get_errors_and_values(
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
            ax=ax
        )

    @staticmethod
    def get_errors_and_values(
            bootstrap_type: Literal['percentile', 'bca'],
            ci_level: float,
            confidence_intervals: bool,
            inferences: List['AbstractInference'],
            labels: list | np.ndarray,
            param_names: list | np.ndarray
    ):
        """
        Get errors and values for inferences.

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
            bootstraps: pd.DataFrame,
            params: dict,
            model: Parametrization,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile'

    ) -> (np.ndarray, np.ndarray):
        """
        Get discretized DFE.

        :param bootstraps: Bootstrap samples
        :param params: Parameters of the model
        :param model: DFE parametrization
        :param bootstrap_type: Type of bootstrap to use
        :param ci_level: Confidence interval level
        :param confidence_intervals: Whether to compute confidence intervals
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bars.
        :return: Values, errors
        """
        if confidence_intervals and bootstraps is not None:
            # get bootstraps and errors if specified
            errors, _, bs, means, values = Inference.get_errors_discretized_dfe(
                params=params,
                bootstraps=bootstraps,
                model=model,
                ci_level=ci_level,
                intervals=intervals,
                bootstrap_type=bootstrap_type
            )
        else:
            # otherwise just get discretized values
            values = Inference.compute_histogram(
                params=params,
                model=model,
                intervals=intervals
            )
            errors, means, bs = None, None, None

        # whether to use the mean of all bootstraps instead of the original values
        use_means = confidence_intervals and bootstraps is not None and bootstrap_type == 'percentile'

        if use_means:
            values = np.mean(bs, axis=0)

        return values, errors

    @staticmethod
    def get_errors_discretized_dfe(
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
        Discretize the DFE given the DFE parametrization and the parameters.

        :param model: DFE parametrization
        :param params: Parameters of the model
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :return: Discretized DFE
        """
        # discrete DFE
        y = from_string(model).discretize(params, intervals)

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
        self.bootstraps: Optional[pd.DataFrame] = None
        self.params_mle: Optional[dict] = None
        self.model: Optional[Parametrization] = None

    @abstractmethod
    def get_bootstrap_params(self) -> dict:
        """
        Get the parameters to be included in the bootstraps.

        :return: Parameters to be included in the bootstraps
        """
        pass

    def get_errors_discretized_dfe(
            self,
            ci_level: float = 0.05,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile'
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """
        Compute errors and confidence interval for a discretized DFE.

        :param ci_level: Confidence interval level
        :param intervals: Array of interval boundaries yielding ``intervals.shape[0] - 1`` bins.
        :param bootstrap_type: Type of bootstrap
        :return: Arrays of errors, confidence intervals, bootstraps, means and values
        """
        return Inference.get_errors_discretized_dfe(
            params=self.get_bootstrap_params(),
            bootstraps=self.bootstraps,
            model=self.model,
            ci_level=ci_level,
            intervals=intervals,
            bootstrap_type=bootstrap_type
        )

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
        :return: Discretized DFE
        """
        return Inference.get_discretized(
            bootstraps=self.bootstraps,
            params=self.get_bootstrap_params(),
            model=self.model,
            intervals=intervals,
            confidence_intervals=confidence_intervals,
            ci_level=ci_level,
            bootstrap_type=bootstrap_type
        )

    def plot_discretized(
            self,
            file: str = None,
            show: bool = True,
            intervals: np.ndarray = np.array([-np.inf, -100, -10, -1, 0, 1, np.inf]),
            confidence_intervals: bool = True,
            ci_level: float = 0.05,
            bootstrap_type: Literal['percentile', 'bca'] = 'percentile',
            title: str = 'discretized DFE',
            ax: plt.Axes = None
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
        :param ax: Axes to plot to
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
            ax=ax
        )

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
