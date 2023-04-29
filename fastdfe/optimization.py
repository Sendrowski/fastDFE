import copy
import logging
from dataclasses import dataclass
from typing import Callable, List, Dict, Literal, Tuple, Optional

import multiprocess as mp
import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import loguniform, uniform
from tqdm import tqdm

from .mle import MLE

# get logger
logger = logging.getLogger('fastdfe')


def parallelize(
        func: Callable,
        data: list | np.ndarray,
        parallelize: bool = True,
        pbar: bool = None
) -> np.ndarray:
    """
    Convenience function that parallelizes the given function
    if specified or executes them sequentially otherwise.

    :param pbar: Whether to show a progress bar
    :param parallelize: Whether to parallelize
    :param data: Data to iterate over
    :param func: Function to apply to each element of data
    :return: List of results
    """
    from . import disable_pbar

    n = len(data)

    if parallelize and n > 1:
        # parallelize
        iterator = mp.Pool().imap(func, data)
    else:
        # sequentialize
        iterator = map(func, data)

    # whether to show a progress bar
    if pbar is True or (pbar is None and not parallelize and n > 1) or pbar is None and n > mp.cpu_count():
        iterator = tqdm(iterator, total=n, disable=disable_pbar)

    return np.array(list(iterator), dtype=object)


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


def unpack_params(x: np.ndarray, original: Dict[str, dict]) -> Dict[str, dict]:
    """
    Unpack params from numpy array. This is the inverse of pack_params and is used
    as scipy.optimize.minimize only accepts numpy arrays as parameters.

    :param x: Numpy array
    :param original: Original dictionary
    :return: Unpacked dictionary
    """
    keys = flatten_dict(original).keys()

    return unflatten_dict(dict(zip(keys, x)))


def pack_params(params: Dict[str, dict]) -> np.ndarray:
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

    :param fixed_params: Dictionary of fixed parameters
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


def correct_values(params: Dict[str, float], bounds: Dict[str, tuple], warn: bool = False) -> Dict[str, float]:
    """
    Correct initial values so that they are within the specified bounds.

    :param bounds: Dictionary of bounds
    :param params: Dictionary of initial values
    :param warn: Whether to warn if values are corrected
    :return: Corrected dictionary
    """
    # create a copy of x0
    corrected = params.copy()

    for key, value in params.items():
        # get base name
        name = key.split('.')[-1]

        if value < bounds[name][0]:
            corrected[key] = bounds[name][0]
        elif value > bounds[name][1]:
            corrected[key] = bounds[name][1]

    if corrected != params and warn:
        logger.warning(f'Given initial values outside bounds. Adjusting {params} to {corrected}.')

    return corrected


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


@dataclass
class SharedParams:
    """
    Class specifying the sharing of params among types.
    'all' means all available types or params.
    """
    #: The params to share
    params: List[str] | Literal['all']

    #: The types to share
    types: List[str] | Literal['all']


@dataclass
class Covariate:
    """
    Class defining a covariate which induces a relationship
    with one or many parameters. The relationship is defined
    by a callback function which modifies the parameters. The
    default callback introduces a linear relationship.

    .. warning::
        TODO let covariate vary on log scale?
    """

    #: The parameter to modify
    param: str

    #: The values of the covariate for each type
    values: Dict[str, float]

    #: The callback function to modify the parameters
    callback: Optional[Callable] = None

    #: The bounds of the covariate parameter to be estimated
    bounds: tuple = (-10, 10)

    #: The initial value of the covariate
    x0: float = 0

    # the scale of the bounds
    bounds_scale: Literal['lin', 'log'] = 'lin'

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


class Optimization:
    """
    Class for optimizing the DFE.
    """

    def __init__(
            self,
            bounds: Dict[str, tuple],
            scales: dict,
            param_names: List[str],
            loss_type: Literal['likelihood', 'L2'] = 'likelihood',
            opts_mle: dict = {},
            parallelize: bool = True,
            fixed_params: Dict[str, Dict[str, float]] = {},
            seed: int = None
    ):
        """
        Create object.

        :param parallelize: Whether to parallelize the optimization
        :param bounds: Dictionary of bounds
        :param opts_mle: Dictionary of options for the optimizer
        :param loss_type: Type of loss function to use
        :param fixed_params: Dictionary of fixed parameters
        :param param_names: List of parameter names
        """
        self.bounds = bounds
        self.scales = scales

        self.opts_mle = opts_mle
        self.loss_type = loss_type

        self.fixed_params = flatten_dict(fixed_params)

        # check if fixed parameters are within the specified bounds
        if correct_values(self.fixed_params, self.bounds, warn=False) != self.fixed_params:
            raise ValueError('Fixed parameters are outside the specified bounds. '
                             f'Fixed params: {self.fixed_params}, bounds: {self.bounds}.')

        self.param_names = param_names
        self.parallelize = parallelize

        # the initial values
        self.x0: Optional[dict] = None

        # the number of runs
        self.n_runs: Optional[int] = None

        # store the likelihoods for each run
        self.likelihoods: Optional[np.ndarray] = None

        # get a random generator instance
        self.rng = np.random.default_rng(seed=seed)

    def run(
            self,
            get_counts: Dict[str, Callable],
            x0: Dict[str, Dict[str, float]] = {},
            n_runs: int = 1,
            debug_iterations: bool = True,
            print_info: bool = True,
            pbar: bool = None
    ) -> (OptimizeResult, dict):
        """
        Perform the optimization procedure.

        :param pbar: Whether to show a progress bar
        :param print_info: Whether to inform the user about the bounds
        :param n_runs: Number of optimization runs
        :param x0: Dictionary of initial values
        :param get_counts: Dictionary of functions to evaluate counts for each type
        :param debug_iterations: Whether to print debug messages for each iteration
        :return: The optimization result and the likelihoods
        """
        # number of optimization runs
        self.n_runs = n_runs

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
        self.x0 = unflatten_dict(correct_values(flattened, self.bounds, warn=True))

        # determine parameter bounds
        bounds = self.get_bounds(flatten_dict(self.x0))

        def optimize(x0: Dict[str, dict]) -> OptimizeResult:
            """
            Perform numerical minimization.

            :param x0: Initial values
            :return: Optimization result
            """
            logger.debug(f"Initial parameters: {x0}.")

            return minimize(
                fun=self.get_loss_function(
                    get_counts=get_counts,
                    print_debug=debug_iterations
                ),
                x0=pack_params(x0),
                method="L-BFGS-B",
                bounds=pack_params(bounds),
                options=self.opts_mle
            )

        # initial parameters for the samples
        initial_params = [self.x0] + [self.sample_x0(self.x0) for _ in range(self.n_runs - 1)]

        # parallelize MLE for different initializations
        results = parallelize(optimize, initial_params, self.parallelize, pbar=pbar)

        # list of the best likelihood for each run
        self.likelihoods = -np.array([res.fun for res in results])

        # get result with the lowest likelihood
        result = results[np.argmax(self.likelihoods)]

        # unpack MLE params array into a dictionary
        params_mle = unpack_params(result.x, self.x0)

        # check if the MLE reached one of the bounds
        if print_info:
            self.check_bounds(flatten_dict(params_mle))

        # unpack shared parameters
        params_mle = unpack_shared(params_mle)

        return result, params_mle

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
            # unpack parameters into dictionary use the keys of self.x0
            params = unpack_params(x, self.x0)

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
            LL = MLE.log_poisson(mu=counts_modelled, k=counts_observed)

            # combine likelihoods and take additive inverse
            ll = np.sum(LL)

            # compute L2 norm
            L2 = norm(counts_modelled - counts_observed, 2)

            # information on iteration
            iter_info = flatten_dict(params) | dict(likelihood=ll, L2=L2)

            # log likelihood
            if print_debug:
                # for setting break points
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
                sample[key] = self.sample_value(self.bounds[key], self.scales[key])

        return sample

    @staticmethod
    def sample_value(bounds: Tuple[float, float], scale: Literal['lin', 'log']) -> float:
        """
        Sample a value between given bounds using the given scaling.
        This function works for positive, negative, and mixed bounds.

        :param bounds: Tuple of lower and upper bounds
        :param scale: Scaling of the parameter.
        :return: Sampled value
        """

        def flip(bounds: Tuple[float, float]) -> Tuple[float, float]:
            return -bounds[1], -bounds[0]

        scaling_functions = {
            'lin': uniform.rvs,
            'log': loguniform.rvs
        }

        # raise an error if the scale is not valid
        if scale not in scaling_functions:
            raise ValueError(f"Scale must be one of: {', '.join(scaling_functions.keys())}")

        # raise an error if bounds span 0 and scale is 'log'
        if bounds[0] < 0 < bounds[1] and scale == 'log':
            raise ValueError('Log scale not possible for bounds that span 0.')

        # flip bounds if they are negative
        flipped = bounds[0] < 0
        if flipped:
            bounds = flip(bounds)

        # sample a value using the appropriate scaling function
        sample = scaling_functions[scale](bounds[0], bounds[1] - bounds[0])

        # return the sampled value, flipping back if necessary
        return -sample if flipped else sample

    def get_bounds(self, x0: dict) -> dict:
        """
        Get a nested dictionary of bounds the same structure as the given initial values.

        :param x0: Initial values
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

    def check_bounds(self, params: Dict[str, float], percentile: float = 1):
        """
        Issue warnings if the passed parameters are close to the specified bounds.

        :param params: The parameters to check.
        :param percentile: The percentile threshold to consider a parameter close to the bounds.
        """
        near_lower = {}
        near_upper = {}

        for key, value in params.items():
            # get base name
            name = key.split('.')[-1]

            # get bounds
            lower, upper = self.bounds[name]

            if key not in self.fixed_params:
                if lower is not None and (value - lower) / (upper - lower) <= percentile / 100:
                    near_lower[key] = lower

                if upper is not None and (upper - value) / (upper - lower) <= percentile / 100:
                    near_upper[key] = upper

        if len(near_lower | near_upper) > 0:
            logger.warning(f'The MLE estimate is within {percentile}% of the upper bound '
                           f'for {near_upper} and lower bound for {near_lower}, but '
                           f'this might be nothing to worry about.')

    def set_fixed_params(self, fixed_params: Dict[str, Dict[str, float]]):
        """
        Set fixed parameters. We flatten the dictionary to make it easier to work with.

        :param fixed_params: Dictionary of fixed parameters
        """
        self.fixed_params = flatten_dict(fixed_params)
