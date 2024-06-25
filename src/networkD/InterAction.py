import inspect

import numpy as np

import System


def _validate_update_function(update_func):
    if update_func:
        updater_signature = inspect.signature(update_func)
        updater_params = updater_signature.parameters
        assert len(updater_params) == 2, "Updater function must accept two parameters: (System, maps)"
        assert updater_signature.return_annotation in [np.ndarray, None], \
            "Updater must return a numpy ndarray or None"


def _validate_map_function(map_func):
    mf_signature = inspect.signature(map_func)
    mf_params = mf_signature.parameters
    assert len(mf_params) == 3, "Map function must accept three parameters: (int, int, System)"


def _check_first_three_params(params, expected_types):
    param_names = list(params)[:3]
    param_types = [params[name].annotation for name in param_names]
    for i, expected_type in enumerate(expected_types):
        assert param_types[i] == expected_type, \
            f"The parameter {param_names[i]} must be of type {expected_type.__name__}"


def _check_keyword_params(params, kwargs):
    for param_name, param in params.items():
        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            assert param_name in kwargs, \
                f"Missing keyword argument: {param_name} required by the rate function"
            if param.default is not inspect.Parameter.empty:
                assert kwargs[param_name] == param.default, \
                    f"Default value for {param_name} does not match"


def _validate_rate_function(rate_func, kwargs):
    rf_signature = inspect.signature(rate_func)
    rf_params = rf_signature.parameters
    _check_first_three_params(rf_params, expected_types=[int, int, System.System])
    _check_keyword_params(rf_params, kwargs)


class InterAction:
    """
    Represents an interaction within a system, encompassing a rate function,
    a map function, and optionally an updater function.

    The rate function determines the probability or rate of interaction between nodes.
    The map function defines how the state of the system changes due to an interaction.
    The updater function, if provided, updates a boolean matrix indicating elements that need updating.
    """

    def __init__(self, *, rate_func, map_func, update_func=None, **kwargs):
        """
        Initializes the interaction with given functions and parameters.
        """
        _validate_rate_function(rate_func, kwargs)
        _validate_map_function(map_func)
        _validate_update_function(update_func)
        self.rate_func = rate_func
        self.map_func = map_func
        self.update_func = update_func
        self.kwargs = kwargs

        self.n_body = 2 # number of bodies
    # Other parts of the class remain unchanged ...

    def get_propensity(self, i: int, j: int, S: System.System):
        """
        Calculate the interaction rate between two nodes.

        :param i: Index of the first node.
        :param j: Index of the second node.
        :param S: The current state of the system.
        :return: The calculated interaction rate.
        """
        if self.kwargs:
            return self.rate_func(i, j, S, **self.kwargs)
        else:
            return self.rate_func(i, j, S)

    def get_map(self, i: int, j: int, S: System.System):
        """
        Apply the map function to the system after an interaction between two nodes.

        :param i: Index of the first node.
        :param j: Index of the second node.
        :param S: The current state of the system.
        :return: The mappings for the new state of the system after the interaction.
        """
        return self.map_func(i, j, S)

    def get_update_matrix(self, S: System.System, maps):
        """
        Update the interaction matrix using the updater function, if provided.

        :param S: The current state of the system.
        :param maps: The interaction maps to be applied.
        :return: A matrix that tells you which elements of the interaction matrix needs updating or None if no updater function is provided.
        """
        if self.update_func:
            result = self.update_func(S, maps)
            assert (isinstance(result, np.ndarray) and result.ndim == 2) or result is None, \
                "Updater function must return a 2D numpy ndarray or None"
            return result
        else:
            return None

