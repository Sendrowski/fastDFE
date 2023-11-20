from testing import prioritize_installed_packages

prioritize_installed_packages()

from typing import Dict
from testing import TestCase

from fastdfe.optimization import Covariate


class CovariateTestCase(TestCase):
    """
    Test the Covariate class.
    """
    def custom_apply_callback(self, covariate: float, type: str, params: Dict[str, float]) -> Dict[str, float]:
        modified = params.copy()
        if "a" in params:
            modified["a"] *= covariate * 2
        return modified

    def test_default_callbacks(self):
        covariate = Covariate(param="a", values={"all": 2})
        params = {"a": 1, "b": 3}

        modified_params = covariate.apply(covariate=2, type="all", params=params)
        self.assertEqual(modified_params, {"a": 5, "b": 3})

    def test_custom_callbacks(self):
        covariate = Covariate(
            param="a",
            values={"all": 2},
            callback=self.custom_apply_callback
        )
        params = {"a": 1, "b": 3}

        modified_params = covariate.apply(covariate=2, type="all", params=params)
        self.assertEqual(modified_params, {"a": 4, "b": 3})
