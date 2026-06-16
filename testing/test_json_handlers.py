import json

import numpy as np

import fastdfe as fd
from fastdfe.json_handlers import CustomEncoder
from fastdfe.optimization import SharedParams, Covariate
from testing import TestCase


class CustomEncoderTestCase(TestCase):
    """
    Fast tests for :class:`fastdfe.json_handlers.CustomEncoder`, the JSON encoder used when writing
    configs/results. It exercises each ``default`` branch (the fastDFE and numpy/scipy types) in
    microseconds, where previously these lines were only hit incidentally by heavy inference runs.
    """

    @staticmethod
    def _encode(obj):
        """Round-trip an object through ``CustomEncoder`` back to a Python value."""
        return json.loads(json.dumps(obj, cls=CustomEncoder))

    def test_encode_spectrum(self):
        """A Spectrum encodes to its list representation."""
        sfs = fd.Spectrum([100, 5, 3, 2, 8])
        self.assertEqual(sfs.to_list(), self._encode(sfs))

    def test_encode_spectra(self):
        """A Spectra encodes to its dict representation."""
        spectra = fd.Spectra(dict(all=[100, 5, 3, 2, 8]))
        self.assertEqual(spectra.to_dict(), self._encode(spectra))

    def test_encode_numpy_array(self):
        """A numpy array encodes to a plain list."""
        self.assertEqual([1.0, 2.0, 3.0], self._encode(np.array([1.0, 2.0, 3.0])))

    def test_encode_numpy_int64(self):
        """A numpy int64 encodes to a plain int."""
        result = self._encode(np.int64(42))
        self.assertEqual(42, result)
        self.assertIsInstance(result, int)

    def test_encode_parametrization(self):
        """A Parametrization encodes to its class name."""
        self.assertEqual('GammaExpParametrization',
                         self._encode(fd.GammaExpParametrization()))

    def test_encode_shared_params(self):
        """SharedParams encodes via its ``__dict__``."""
        shared = SharedParams(types='all', params=['S_d'])
        self.assertEqual(['S_d'], self._encode(shared)['params'])

    def test_encode_covariate(self):
        """A Covariate encodes to just its ``param`` and ``values``."""
        cov = Covariate(param='S_d', values=dict(a=0.3, b=0.6))
        self.assertEqual({'param': 'S_d', 'values': {'a': 0.3, 'b': 0.6}}, self._encode(cov))

    def test_encode_lbfgs_inv_hess_product(self):
        """The L-BFGS inverse-Hessian object (which resists pickling) encodes to its string form."""
        from scipy.optimize._lbfgsb_py import LbfgsInvHessProduct

        obj = LbfgsInvHessProduct(np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))
        self.assertEqual(str(obj), self._encode(obj))

    def test_unsupported_type_raises(self):
        """An unsupported type falls through to the base encoder and raises ``TypeError``."""
        with self.assertRaises(TypeError):
            json.dumps({1, 2, 3}, cls=CustomEncoder)
