"""
Optimization module.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-02-26"

import copy
import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Dict, Literal, Tuple, Optional

import multiprocess as mp
import numpy as np
from numpy.linalg import norm
from numpy.random import Generator
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import loguniform, uniform
from tqdm import tqdm

from .likelihood import Likelihood
from .settings import Settings

# get logger
logger = logging.getLogger('fastdfe').getChild('Optimization')


def parallelize(
        func: Callable,
        data: list | np.ndarray,
        parallelize: bool = True,
        pbar: bool = None,
        desc: str = None,
        dtype: type = object
) -> np.ndarray:
    """
    Parallelize given function or execute sequentially.

    :param parallelize: Whether to parallelize
    :param data: Data to iterate over
    :param func: Function to apply to each element of data
    :param pbar: Whether to show a progress bar
    :param desc: Description for progress bar
    :param dtype: Data type of the returned array
    :return: List of results
    """
    n = len(data)

    if parallelize and n > 1:
        # parallelize
        iterator = mp.Pool().imap(func, data)
    else:
        # sequentialize
        iterator = map(func, data)

    # whether to show a progress bar
    if pbar is True or (pbar is None and not parallelize and n > 1) or pbar is None and n > mp.cpu_count():
        iterator = tqdm(iterator, total=n, disable=Settings.disable_pbar, desc=desc)

    return np.array(list(iterator), dtype=dtype)


def flatten_dict(d: dict, separator='.', prefix=''):
    """
    Flatten dictionary.

    :param d: The nested dictionary
    :param separator: The separator character to use in the flattened dictionary keys
    :param prefix: The prefix to use in the flattened dictionary keys
    :return: The flattened dictionary
    """
    res = {}

    for key, value in d.items():
        if isinstance(value, dict):
            # recursive call
            res.update(flatten_dict(value, separator, prefix + key + separator))
        else:
            res[prefix + key] = value

    return res


def unflatten_dict(d: dict, separator='.'):
    """
    Unflatten dictionary.

    :param d: The flattened dictionary
    :param separator: The separator character used in the flattened dictionary keys
    :return: The original nested dictionary
    """
    res = {}

    for key, value in d.items():
        subkeys = key.split(separator)
        subdict = res

        # recursively create nested dictionaries for each subkey
        for subkey in subkeys[:-1]:
            subdict = subdict.setdefault(subkey, {})

        # assign value to the final subkey
        subdict[subkeys[-1]] = value if not isinstance(value, dict) else unflatten_dict(value, separator)

    return res


def unpack_params(x: np.ndarray, original: Dict[str, dict | tuple | float]) -> Dict[str, dict | tuple | float]:
    """
    Unpack params from numpy array. This is the inverse of pack_params and is used
    as scipy.optimize.minimize only accepts numpy arrays as parameters.

    :param x: Numpy array
    :param original: Original dictionary
    :return: Unpacked dictionary
    """
    keys = flatten_dict(original).keys()

    return unflatten_dict(dict(zip(keys, x)))


def pack_params(params: Dict[str, dict | tuple | float]) -> np.ndarray:
    """
    Pack params into numpy array. This is used as scipy.optimize.minimize only accepts
    numpy arrays as parameters. This is the inverse of unpack_params.

    :param params: Dictionary to pack
    :return: numpy array
    """
    flattened = flatten_dict(params)

    return np.array(list(flattened.values()))


def filter_dict(d, keys):
    """
    Recursively filter a dictionary by a list of given keys at the deepest level.

    :param d: The dictionary to filter
    :param keys: The list of keys to keep in the filtered dictionary
    :return: The filtered dictionary
    """
    filtered = {}

    for key, value in d.items():
        if isinstance(value, dict):
            filtered_sub = filter_dict(value, keys)
            if filtered_sub:
                filtered[key] = filtered_sub
        else:
            if key in keys:
                filtered[key] = value

    return filtered


def pack_shared(
        params: Dict[str, Dict[str, float]],
        shared: List['SharedParams'],
        shared_values: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Pack shared parameters. Here we extract shared parameters from
    type-specific keys and instead create entries with joint-types
    holding the shared parameters.
    Note that we only delete parameters from marginal types but in some
    cases we would have to delete them from compound types as well if
    a more general shared parameter were to be specified later on.
    We instead rely on the user not to do this.

    :param params: Dictionary of parameters indexed by type
    :param shared: List of shared parameters
    :param shared_values: Dictionary of parameters indexed by joint-type and parameter name, i.e. 'type1:type2.p1'
    :return: Packed dictionary
    """
    packed = copy.deepcopy(params)

    # iterator through shared parameter
    for s in shared:

        # remove shared keys
        for t in s.types:
            for p in s.params:
                if p in packed[t]:
                    packed[t].pop(p)

        # create joint type string
        type_string = ':'.join(s.types)

        # create shared type with appropriate parameters
        packed = merge_dicts(packed, {type_string: dict((p, shared_values[type_string + '.' + p]) for p in s.params)})

    return packed


def unpack_shared(params: dict) -> dict:
    """
    Unpack shared parameters. Here we extract shared parameters from joint-type keys
    and add them tp the targeted types.

    :param params: Dictionary of parameters
    :return: Unpacked dictionary
    """
    unpacked = {}

    for t, v in params.items():
        if ':' in t:
            unpacked = merge_dicts(unpacked, dict((s, v) for s in t.split(':')))
        else:
            unpacked[t] = v

    return unpacked


def expand_shared(params: List['SharedParams'], types: List[str], names: List[str]) -> List['SharedParams']:
    """
    Expand 'all' type for shared parameters.

    :param params: List of shared parameters
    :param types: List of types
    :param names: List of parameter names
    :return: Expanded list of shared parameters
    """
    expanded = []

    for x in params:
        # expand 'all' type
        types = types if x.types == 'all' else x.types
        params = names if x.params == 'all' else x.params

        # noinspection PyTypeChecker
        expanded.append(SharedParams(types=types, params=params))

    return expanded


def expand_fixed(
        fixed_params: Dict[str, Dict[str, float]],
        types: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Expand 'all' type for shared parameters.

    :param fixed_params: Dictionary of fixed parameters indexed by type and parameter
    :param types: List of types
    :return: Expanded dictionary of fixed parameters
    """
    expanded = {}

    # loop through fixed parameters
    for key_type, params in fixed_params.items():
        # expand 'all' type
        key_types = types if key_type == 'all' else [key_type]

        # loop through types
        for t in key_types:
            if t not in expanded:
                expanded[t] = {}

            for param, value in params.items():
                expanded[t][param] = value

    return expanded


def collapse_fixed(
        expanded_params: Dict[str, Dict[str, float]],
        types: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Collapse expanded fixed parameters to 'all' type if all types have the same fixed parameter.

    :param expanded_params: Expanded dictionary of fixed parameters
    :param types: List of types
    :return: Collapsed dictionary of fixed parameters
    """
    all_params = {param: [] for params in expanded_params.values() for param in params}

    # Collect parameter values for all types
    for params in expanded_params.values():
        for param, value in params.items():
            all_params[param].append(value)

    # Calculate the mean and check if parameters can be collapsed
    collapsed = {}
    for param, values in all_params.items():
        if len(values) == len(types):
            collapsed[param] = np.mean(values)

    return {'all': collapsed} if len(collapsed) > 0 else expanded_params


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    """
    Merge two dictionaries recursively.

    :param dict1: First dictionary
    :param dict2: Second dictionary
    :return: Merged dictionary
    """
    # make a copy of the first dictionary
    result = dict(dict1)

    # loop through the items in the second dictionary
    for key, value in dict2.items():

        # Check if the key already exists in the result dictionary and both the
        # value in the result and dict2 dictionaries are dictionaries.
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):

            # recursively merge the two dictionaries
            result[key] = merge_dicts(result[key], value)
        else:
            # simply assign the value from dict2 to the result dictionary
            result[key] = value

    return result


def correct_values(
        params: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
        scales: Dict[str, Literal['lin', 'log', 'symlog']],
        warn: bool = False,
        threshold: float = 1e-6
) -> Dict[str, float]:
    """
    Correct initial values so that they are within the specified bounds.

    :param bounds: Dictionary of bounds
    :param params: Flattened dictionary of parameters
    :param scales: Dictionary of scales
    :param warn: Whether to warn if values are corrected
    :param threshold: Threshold for the error to trigger a warning
    :return: Corrected dictionary
    """
    # create a copy of params
    corrected = params.copy()

    for key, value in params.items():
        # get base name
        name = key.split('.')[-1]

        # get real bounds
        bound = get_real_bounds(bounds[name], scale=scales[name])

        # correct value if outside bounds
        if value < bound[0]:
            corrected[key] = bound[0]
        elif value > bound[1]:
            corrected[key] = bound[1]

    # differences between the original and corrected dictionaries
    differences = {key: (params[key], corrected[key]) for key in params if params[key] != corrected[key]}

    # warn if there are differences that exceed the threshold
    exceeded_threshold = {}
    for key, (old_val, new_val) in differences.items():

        # calculate relative error
        err = np.abs(new_val - old_val)

        # add if it exceeds the relative error
        if err > threshold:
            exceeded_threshold[key] = f"{old_val} -> {new_val}"

    if exceeded_threshold and warn:
        logger.warning(f'Given initial values outside bounds. Adjusting {exceeded_threshold}.')

    return corrected


def get_real_bounds(bounds: Tuple[float, float], scale: Literal['lin', 'log', 'symlog']) -> Tuple[float, float]:
    """
    Get real bounds from the given bounds.

    :param bounds: Bounds of the parameter
    :param scale: Scale of the parameter
    :return:
    """
    if scale == 'symlog':
        return -bounds[1], bounds[1]

    return bounds


def evaluate_counts(get_counts: dict, params: dict):
    """
    Evaluate counts using the given parameters.
    Here we assign the parameters to the appropriate types
    obtaining the counts for each type.

    :param get_counts: Dictionary of functions to evaluate counts for each type
    :param params: Dictionary of parameters
    :return: Dictionary of counts
    """
    counts = {}

    # unpack shared parameters
    unpacked = unpack_shared(params)

    # evaluate counts for each type
    for key in get_counts.keys():
        counts[key] = get_counts[key](unpacked[key])

    return counts


def to_symlog(x: float, linthresh: float = 1e-5) -> float:
    """
    Convert a value to the symlog scale.

    :param x: The input value on the original scale.
    :param linthresh: The positive value that determines the range within which the
        symlog scale is linear. Must be greater than 0.
    :return: The value on the symlog scale.
    """
    sign = np.sign(x)
    abs_x = np.abs(x)
    log_x = np.log10(abs_x + linthresh) - np.log10(linthresh)

    return sign * (abs_x / linthresh if abs_x <= linthresh else log_x)


def from_symlog(y: float, linthresh: float = 1e-5) -> float:
    """
    Convert a value from the symlog scale back to the original scale.

    :param y: The input value on the symlog scale.
    :param linthresh: The positive value that determines the range within which the
        symlog scale is linear. Must be greater than 0.
    :return: The value on the original scale.
    """
    sign = np.sign(y)
    abs_y = np.abs(y)
    exp_y = np.power(10, abs_y + np.log10(linthresh)) - linthresh

    return sign * (abs_y * linthresh if abs_y <= 1 else exp_y)


def scale_bound(bounds: Tuple[float, float], scale: Literal['lin', 'log', 'symlog']):
    """
    Convert a bound to the specified scale. For symlog scale we assume the symmetric bounds,
    so that the upper bound denotes the boundaries and the lower bound the linear threshold.

    :param bounds: The bound to convert
    :param scale: The scale to convert to
    :return: The converted bound
    :raises ValueError: if the scale is unknown
    """
    if scale == 'lin':
        return bounds

    if scale == 'log':
        if bounds[1] < 0:
            return -np.log10(-bounds[0]), -np.log10(-bounds[1])

        if bounds[0] > 0:
            return np.log10(bounds[0]), np.log10(bounds[1])

        raise ValueError('Bounds must not span zero for log scale.')

    if scale == 'symlog':

        if bounds[0] <= 0 or bounds[1] <= 0:
            raise ValueError('Both bounds must be positive for symlog scale.')

        return to_symlog(-bounds[1], linthresh=bounds[0]), to_symlog(bounds[1], linthresh=bounds[0])

    raise ValueError(f'Unknown scale {scale}.')


def unscale_bound(
        scaled_bounds: Tuple[float, float],
        scale: Literal['lin', 'log', 'symlog'],
        linthresh: float = 1e-5
) -> Tuple[float, float]:
    """
    Convert a bound from the specified scale back to the original scale. For symlog scale,
    we assume symmetric bounds, so that the upper bound denotes the boundaries and
    the lower bound the linear threshold, i.e. ``bounds = (-bounds[1], bounds[1])`` and ``linthresh = bounds[0]``.
    Note that we cannot reliably recover negative bounds that were log scaled.

    :param linthresh:
    :param scaled_bounds: The bound to convert
    :param scale: The scale to convert from
    :return: The converted bound
    :raises ValueError: if the scale is unknown
    """
    if scale == 'lin':
        return scaled_bounds

    if scale == 'log':
        return np.power(10, scaled_bounds[0]), np.power(10, scaled_bounds[1])

    if scale == 'symlog':
        upper_bound = from_symlog(scaled_bounds[1], linthresh=linthresh)

        return linthresh, upper_bound

    raise ValueError(f'Unknown scale {scale}.')


def scale_bounds(
        bounds: Dict[str, Tuple[float, float]],
        scales: Dict[str, Literal['lin', 'log', 'symlog']]
) -> Dict[str, Tuple[float, float]]:
    """
    Convert bounds to the specified scale. For symlog scale we assume the symmetric bounds,
    so that the upper bound denotes the boundaries and the lower bound the linear threshold.

    :param bounds: Flattened dictionary of bounds to convert index by type and parameter
    :param scales: Dictionary of scales indexed by parameter
    :return: The converted bounds
    :raises ValueError: if the scale is unknown
    """
    scaled_bounds = {}

    for key, value in bounds.items():
        scaled_bounds[key] = scale_bound(value, scale=scales[get_basename(key)])

    return scaled_bounds


def scale_value(value: float, bounds: Tuple[float, float], scale: Literal['lin', 'log', 'symlog']) -> float:
    """
    Convert a value to the specified scale. For symlog scale, the untransformed bounds are needed,
    so that the upper bound denotes the boundaries and the lower bound the linear threshold.

    :param value: The value to convert.
    :param bounds: The untransformed bounds for the symlog scale.
    :param scale: The scale to convert to.
    :return: The converted value.
    :raises ValueError: if the scale is unknown.
    """
    if scale == 'lin':
        return value

    if scale == 'log':

        if value < 0:
            return -np.log10(-value)

        return np.log10(value)

    if scale == 'symlog':
        return to_symlog(value, linthresh=bounds[0])

    raise ValueError(f'Unknown scale {scale}.')


def unscale_value(scaled_value: float, bounds: Tuple[float, float], scale: Literal['lin', 'log', 'symlog']) -> float:
    """
    Convert a value from the specified scale back to the original scale. For symlog scale,
    the untransformed bounds are needed, so that the upper bound denotes the boundaries
    and the lower bound the linear threshold.

    :param scaled_value: The value to convert.
    :param bounds: The untransformed bounds for the symlog scale.
    :param scale: The scale to convert from.
    :return: The converted value.
    :raises ValueError: if the scale is unknown.
    """
    if scale == 'lin':
        return scaled_value

    if scale == 'log':

        if bounds[1] < 0:
            return -np.power(10, -scaled_value)

        return np.power(10, scaled_value)

    if scale == 'symlog':
        return from_symlog(scaled_value, linthresh=bounds[0])

    raise ValueError(f'Unknown scale {scale}.')


def scale_values(
        params: Dict[str, Dict[str, float]],
        bounds: Dict[str, Tuple[float, float]],
        scales: Dict[str, Literal['lin', 'log', 'symlog']]
) -> Dict[str, Dict[str, float]]:
    """
    Scale values according to the given scales.

    :param params: Nested dictionary of parameters indexed by type and parameter
    :param scales: Dictionary of scales indexed by parameter name
    :param bounds: Dictionary of bounds indexed by parameter name
    :return: Nested dictionary of scaled parameters indexed by type and parameter
    """
    scaled = {}

    for key, value in flatten_dict(params).items():
        # scale value
        scaled[key] = scale_value(value, bounds[get_basename(key)], scales[get_basename(key)])

    return unflatten_dict(scaled)


def unscale_values(
        params: Dict[str, Dict[str, float]],
        bounds: Dict[str, Tuple[float, float]],
        scales: Dict[str, Literal['lin', 'log', 'symlog']]
) -> Dict[str, Dict[str, float]]:
    """
    Unscale values according to the given scales.

    :param params: Nested dictionary of parameters indexed by type and parameter
    :param scales: Dictionary of scales indexed by parameter name
    :param bounds: Dictionary of scales indexed by parameter name
    :return: Nested dictionary of unscaled parameters indexed by type and parameter
    """
    unscaled = {}

    for key, value in flatten_dict(params).items():
        # unscale value
        unscaled[key] = unscale_value(value, bounds[get_basename(key)], scales[get_basename(key)])

    return unflatten_dict(unscaled)


def get_basename(name: str) -> str:
    """
    Get the basename of parameter string, i.e. type.param -> param.

    :param name: The string to get the basename from.
    :return: The basename.
    """
    return name.split('.')[-1]


def check_bounds(
        bounds: Dict[str, Tuple[float, float]],
        params: Dict[str, float],
        fixed_params: Dict[str, float] = {},
        percentile: float = 1,
        scale: Literal['lin', 'log'] = 'lin'
) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, Tuple[float, float, float]]]:
    """
    Issue warnings if the passed parameters are close to the specified bounds.

    :param bounds: The bounds to check against.
    :param params: The parameters to check.
    :param fixed_params: The fixed parameters.
    :param percentile: The percentile threshold to consider a parameter close to the bounds.
    :param scale: Scale type: 'lin' for linear and 'log' for logarithmic.
    :return: Tuple of dictionaries of parameters close to the lower and upper bounds, i.e. (lower, value, upper).
    """
    near_lower = {}
    near_upper = {}

    def transform(value: float, to_scale: Literal['lin', 'log']) -> float:
        """
        Transform a value to the specified scale.

        :param value: The value to transform.
        :param to_scale: The scale to transform to.
        :return: The transformed value.
        """
        if to_scale == 'log':
            return math.log(value) if value > 0 else -float('inf')

        return value

    for key, value in params.items():
        # get base name
        name = key.split('.')[-1]

        # get bounds
        lower, upper = bounds[name]

        # transform values
        _lower = transform(lower, scale)
        _upper = transform(upper, scale)
        _value = transform(value, scale)

        if key not in fixed_params:
            if _lower is not None and (_value - _lower) / (_upper - _lower) <= percentile / 100:
                near_lower[key] = (lower, value, upper)

            if _upper is not None and (_upper - _value) / (_upper - _lower) <= percentile / 100:
                near_upper[key] = (lower, value, upper)

    return near_lower, near_upper


@dataclass
class SharedParams:
    """
    Class specifying the sharing of params among types.
    'all' means all available types or params.

    Example usage:

    ::

        import fastdfe as fd

        # neutral SFS for two types
        sfs_neut = fd.Spectra(dict(
            pendula=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 652],
            pubescens=[172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31]
        ))

        # selected SFS for two types
        sfs_sel = fd.Spectra(dict(
            pendula=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 794],
            pubescens=[791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41]
        ))

        # create inference object
        inf = fd.JointInference(
            sfs_neut=sfs_neut,
            sfs_sel=sfs_sel,
            shared_params=[fd.SharedParams(types=["pendula", "pubescens"], params=["eps", "S_d"])],
            do_bootstrap=True
        )

        # run inference
        inf.run()

    """
    #: The params to share
    params: List[str] | Literal['all'] = 'all'

    #: The types to share
    types: List[str] | Literal['all'] = 'all'


@dataclass
class Covariate:
    """
    Class defining a covariate which induces a relationship
    with one or many parameters. The relationship is defined
    by a callback function which modifies the parameters. The
    default callback introduces a linear relationship.
    """

    #: The parameter to modify
    param: str

    #: The values of the covariate for each type
    values: Dict[str, float]

    #: The callback function to modify the parameters
    callback: Optional[Callable] = None

    #: The bounds of the covariate parameter to be estimated
    bounds: tuple = (1e-4, 1e4)

    #: The initial value of the covariate
    x0: float = 0

    #: The scale of the bounds. See :func:`scale_value` for details
    bounds_scale: Literal['lin', 'log', 'symlog'] = 'symlog'

    def __post_init__(self):
        """
        Cast bounds to tuple and check if an inverse_callback is provided
        when a custom callback is specified.
        """
        self.bounds = tuple(self.bounds)

    def apply(self, covariate: float, type: str, params: Dict[str, float]) -> Dict[str, float]:
        """
        Apply the custom or default callback to modify the given parameters.

        :param covariate: The value of the covariate.
        :param type: The type of the relationship.
        :param params: The input parameters.
        :return: Modified parameters.
        """
        # Use custom callback if given else default callback
        callback = self.apply_default if self.callback is None else self.callback

        return callback(covariate=covariate, type=type, params=params)

    def apply_default(self, covariate: float, type: str, params: Dict[str, float]) -> Dict[str, float]:
        """
        Modify the given parameters introducing a linear relationship
        with the given covariate.

        :param covariate: The value of the covariate.
        :param type: The type of the relationship.
        :param params: The input parameters.
        :return: Modified parameters.
        """
        # create a copy of input parameters
        modified = params.copy()

        # introduce linear relationship
        if self.param in params:
            modified[self.param] += covariate * self.values[type]

        return modified

    @staticmethod
    def _apply(covariates: Dict[str, 'Covariate'], params: dict, type: str) -> dict:
        """
        Apply given covariates to given parameters.

        :param covariates: Dictionary of covariates to add
        :param params: Dict of parameters
        :param type: SFS type
        :return: Dict of parameters with covariates added
        """
        for k, cov in covariates.items():
            params = cov.apply(
                covariate=params[k],
                type=type,
                params=params
            )

        return params


class Optimization:
    """
    Class for optimizing the DFE.
    """

    def __init__(
            self,
            bounds: Dict[str, Tuple[float, float]],
            param_names: List[str],
            loss_type: Literal['likelihood', 'L2'] = 'likelihood',
            opts_mle: dict = {},
            parallelize: bool = True,
            fixed_params: Dict[str, Dict[str, float]] = {},
            scales: Dict[str, Literal['lin', 'log', 'symlog']] = {},
            seed: int = None
    ):
        """
        Create object.

        :param parallelize: Whether to parallelize the optimization
        :param bounds: Dictionary of bounds
        :param opts_mle: Dictionary of options for the optimizer
        :param loss_type: Type of loss function to use
        :param fixed_params: Dictionary of fixed parameters
        :param scales: Dictionary of scales
        :param param_names: List of parameter names
        """
        #: Parameter bounds
        self.bounds = bounds

        #: Parameter scales to use
        self.scales = scales

        #: additional options for the optimizer
        self.opts_mle = opts_mle

        #: Type of loss function to use
        self.loss_type = loss_type

        #: Fixed parameters
        self.fixed_params = flatten_dict(fixed_params)

        # check if fixed parameters are within the specified bounds
        if correct_values(self.fixed_params, self.bounds, warn=False, scales=scales) != self.fixed_params:
            raise ValueError('Fixed parameters are outside the specified bounds. '
                             f'Fixed params: {self.fixed_params}, bounds: {self.bounds}.')

        #: Parameter names
        self.param_names = param_names

        #: Whether to parallelize the optimization
        self.parallelize = parallelize

        #: Initial values
        self.x0: Optional[dict] = None

        #: Number of runs
        self.n_runs: Optional[int] = None

        #: Likelihoods for each run
        self.likelihoods: Optional[np.ndarray] = None

        #: Random generator instance
        self.rng = np.random.default_rng(seed=seed)

    def run(
            self,
            get_counts: Dict[str, Callable],
            x0: Dict[str, Dict[str, float]] = {},
            scales: Dict[str, Literal['lin', 'log', 'symlog']] = {},
            bounds: Dict[str, Tuple[float, float]] = {},
            n_runs: int = 1,
            debug_iterations: bool = True,
            print_info: bool = True,
            opts_mle: dict = None,
            pbar: bool = None,
            desc: str = 'Inferring DFE',
    ) -> (OptimizeResult, dict):
        """
        Perform the optimization procedure.

        :param scales: Scales of the parameters
        :param bounds: Bounds of the parameters
        :param n_runs: Number of independent optimization runs out of which the best one is chosen. The first run
            will use the initial values if specified. Consider increasing this number if the optimization does not
            produce good results.
        :param x0: Dictionary of initial values in the form ``{type: {param: value}}``
        :param get_counts: Dictionary of functions to evaluate counts for each type
        :param debug_iterations: Whether to print debug messages for each iteration
        :param opts_mle: Dictionary of options for the optimizer
        :param print_info: Whether to print information about the bounds
        :param pbar: Whether to show a progress bar
        :param desc: Description for the progress bar
        :return: The optimization result and the likelihoods
        """
        # number of optimization runs
        self.n_runs = n_runs

        # store the scales of the parameters
        if scales:
            self.scales = scales

        # store the bounds of the parameters
        if bounds:
            self.bounds = bounds

        # store the options for the optimizer
        if opts_mle:
            self.opts_mle = opts_mle

        # filter out unneeded values
        # this also holds the fixed parameters
        self.x0 = filter_dict(x0, self.param_names)

        # flatten initial values
        flattened = flatten_dict(self.x0)

        # determine parameter names of parameters to be optimized
        optimized_param_names = list(set(flattened) - set(self.fixed_params))

        # issue debug messages
        logger.debug(f'Performing optimization on {len(flattened)} parameters: {list(flattened.keys())}.')
        logger.debug(f'Using initial values: {flattened}.')
        logger.debug(f"Optimizing parameters: {optimized_param_names}")

        # issue warning when the number of parameters to be optimized is large
        if len(optimized_param_names) > 10 and print_info:
            logger.warning(f'A large number of parameters is optimized jointly ({len(optimized_param_names)}). '
                           f'Please be aware that this makes it harder to find a good optimum.')

        # correct initial values to be within bounds
        self.x0 = unflatten_dict(correct_values(flattened, self.bounds, warn=True, scales=self.scales))

        # determine parameter bounds
        bounds = self.get_bounds(flatten_dict(self.x0))

        def optimize(x0: Dict[str, Dict[str, float]]) -> OptimizeResult:
            """
            Perform numerical minimization.

            :param x0: Dictionary of initial values in the form ``{type: {param: value}}``
            :return: Optimization result
            """
            logger.debug(f"Initial parameters: {x0}.")

            return minimize(
                fun=self.get_loss_function(
                    get_counts=get_counts,
                    print_debug=debug_iterations
                ),
                x0=pack_params(self.scale_values(x0)),
                method="L-BFGS-B",
                bounds=pack_params(scale_bounds(bounds, self.scales)),
                options=self.opts_mle
            )

        # initial parameters for the samples
        initial_params = [self.x0] + [self.sample_x0(self.x0) for _ in range(int(self.n_runs) - 1)]

        # parallelize MLE for different initializations
        results = parallelize(optimize, initial_params, self.parallelize, pbar=pbar, desc=desc)

        # list of the best likelihood for each run
        self.likelihoods = -np.array([res.fun for res in results])

        # get result with the lowest likelihood
        result = results[np.argmax(self.likelihoods)]

        # unpack MLE params array into a dictionary
        params_mle = unpack_params(result.x, self.x0)

        # unscale parameters
        params_mle = unscale_values(params_mle, self.bounds, self.scales)

        # check if the MLE reached one of the bounds
        if print_info:
            self.check_bounds(flatten_dict(params_mle))

        return result, params_mle

    def scale_values(self, values: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Scale the values of the parameters.

        :param values: Dictionary of initial values in the form ``{type: {param: value}}``
        :return: Dictionary of scaled initial values
        """
        return scale_values(values, self.bounds, self.scales)

    def get_loss_function(
            self,
            get_counts: Dict[str, Callable],
            print_debug: bool = True
    ) -> Callable:
        """
        Get the loss function.

        :param get_counts: Dictionary of functions to evaluate counts for each type
        :param print_debug: Whether to print debug messages
        :return: The loss function
        """

        def loss(x: np.ndarray) -> float:
            """
            The loss function.

            :param x: Parameters
            :return: The loss
            """
            # unpack parameters into dictionary using the keys of self.x0
            params = unpack_params(x, self.x0)

            # unscale parameters
            params = unscale_values(params, self.bounds, self.scales)

            # Model SFS from parameters.
            # Here the order of types does not matter.
            # We only collect the counts for types that are
            # given in get_counts. This makes it possible to
            # avoid specifying type 'all' which is of no use
            # in joint inference.
            counts_dict = evaluate_counts(get_counts, params)

            # flatten and convert to array
            counts = np.array(list(counts_dict.values()))

            # reshape and merge
            counts_modelled, counts_observed = np.stack(counts, axis=1).reshape(2, -1)

            # use independent Poisson likelihoods
            LL = Likelihood.log_poisson(mu=counts_modelled, k=counts_observed)

            # combine likelihoods
            ll = np.sum(LL)

            # compute L2 norm
            L2 = norm(counts_modelled - counts_observed, 2)

            # information on iteration
            iter_info = flatten_dict(params) | dict(likelihood=ll, L2=L2)

            # log likelihood
            if print_debug:
                # check likelihood
                if np.isnan(ll):
                    raise ValueError('Oh boy, likelihood is nan. This is no good...')

                # log variables
                logger.debug(iter_info)

            # return appropriate loss
            return dict(L2=L2, likelihood=-ll)[self.loss_type]

        return loss

    def sample_x0(self, example: dict) -> Dict[str, dict]:
        """
        Sample initial values.

        :param example: An example dictionary for generating the initial values
        :return: A dictionary of initial values
        """
        sample = {}

        for key, value in example.items():
            if isinstance(value, dict):
                sample[key] = self.sample_x0(value)
            else:
                sample[key] = self.sample_value(self.bounds[key], self.scales[key], random_state=self.rng)

        return sample

    @staticmethod
    def sample_value(
            bounds: Tuple[float, float],
            scale: Literal['lin', 'log', 'symlog'],
            random_state: int | Generator = None
    ) -> float:
        """
        Sample a value between given bounds using the given scaling.
        This function works for positive, negative, and mixed bounds.
        Note that when ``scale == 'symlog'``, ``bounds[0]`` defines the linear threshold and
        the actual bounds are ``(-bounds[1], bounds[1])``.

        :param bounds: Tuple of lower and upper bounds
        :param scale: Scaling of the parameter.
        :param random_state: Random state
        :return: Sampled value
        """

        def flip(bounds: Tuple[float, float]) -> Tuple[float, float]:
            """
            Flip the bounds.

            :param bounds: Tuple of lower and upper bounds
            :return: Flipped bounds
            """
            return -bounds[1], -bounds[0]

        def symlog_rvs(lower: float, upper: float, random_state: int | Generator = None) -> float:
            """
            Sample from a symmetric log-uniform distribution.

            :param lower: Lower bound which is the linear threshold
            :param upper: Upper bound so that the actual bounds are (-upper, upper)
            :param random_state: Random state
            :return: Sampled value
            """
            val = loguniform.rvs(lower, upper, random_state=random_state)

            # flip sign with 50% probability
            return val if uniform.rvs() < 0.5 else -val

        # dictionary of scaling functions
        scaling_functions = {
            'lin': uniform.rvs,
            'log': loguniform.rvs,
            'symlog': symlog_rvs
        }

        # raise an error if the scale is not valid
        if scale not in scaling_functions:
            raise ValueError(f"Scale must be one of: {', '.join(scaling_functions.keys())}")

        # raise an error if bounds span 0 and scale is 'log'
        if bounds[0] < 0 < bounds[1] and scale == 'log':
            raise ValueError(f"Log scale not possible for bounds that span 0.")

        # raise an error if bounds are negative and scale is 'symlog'
        if bounds[0] < 0 and scale == 'symlog':
            raise ValueError(f"Symlog scale not possible for negative bounds.")

        # flip bounds if they are negative
        flipped = bounds[0] < 0
        if flipped:
            bounds = flip(bounds)

        # sample a value using the appropriate scaling function
        sample = scaling_functions[scale](bounds[0], bounds[1] - bounds[0], random_state=random_state)

        # return the sampled value, flipping back if necessary
        return -sample if flipped else sample

    def check_bounds(self, params: Dict[str, float], percentile: float = 1) -> None:
        """
        Check if the given parameters are within the bounds.

        :param params: Parameters
        :param percentile: Percentile of the bounds to check
        :return: Whether the parameters are within the bounds
        """
        # we scale the bounds to obtain more sensible warnings
        bounds = scale_bounds(self.bounds, self.scales)
        params_scaled = flatten_dict(self.scale_values(unflatten_dict(params)))

        # get parameters close to the bounds
        near_lower, near_upper = check_bounds(
            params=params_scaled,
            bounds=bounds,
            fixed_params=self.fixed_params,
            percentile=percentile,
            scale='lin'
        )

        if len(near_lower | near_upper) > 0:

            def get_values(keys: List[str]) -> Dict[str, Tuple[str, str, str]]:
                """
                Unscale the parameters.

                :param keys: List of parameter names
                :return: Unscaled parameters
                """
                unscaled = {}

                for key in keys:
                    unscaled[key] = (
                        "{:g}".format(self.bounds[get_basename(key)][0]),
                        "{:.8g}".format(params[key]),
                        "{:g}".format(self.bounds[get_basename(key)][1])
                    )

                return unscaled

            # string representation of parameters
            near_lower_unscaled = str(get_values(list(near_lower.keys()))).replace('\'', '')
            near_upper_unscaled = str(get_values(list(near_upper.keys()))).replace('\'', '')

            # issue warning
            logger.warning(
                f'The MLE estimate is close to the upper bound '
                f'for {near_upper_unscaled} and lower bound '
                f'for {near_lower_unscaled} [(lower, value, upper)], but '
                f'this might be nothing to worry about.'
            )

    def get_bounds(self, x0: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """
        Get a nested dictionary of bounds the same structure as the given initial values.

        :param x0: Flattened dictionary of initial values
        :return: A dictionary of initial values
        """
        bounds = {}

        for key, value in x0.items():

            # check if the parameter is fixed
            if key in self.fixed_params:
                bounds[key] = (self.fixed_params[key], self.fixed_params[key])
            else:
                bounds[key] = self.bounds[key.split('.')[-1]]

        return bounds

    def set_fixed_params(self, fixed_params: Dict[str, Dict[str, float]]):
        """
        Set fixed parameters. We flatten the dictionary to make it easier to work with.

        :param fixed_params: Dictionary of fixed parameters
        """
        self.fixed_params = flatten_dict(fixed_params)
