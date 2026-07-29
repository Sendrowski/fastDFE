.. _modules.base_inference:

DFE Inference
-------------

Classes for inferring the distribution of fitness effects (DFE) from site-frequency
spectra, spanning single-type inference, joint inference across multiple types, and
the shared-parameter and covariate machinery that links them.

**Classes:**

.. autosummary::
   :nosignatures:

   ~fastdfe.base_inference.BaseInference
   ~fastdfe.joint_inference.JointInference
   ~fastdfe.abstract_inference.Inference
   ~fastdfe.base_inference.InferenceResult
   ~fastdfe.optimization.SharedParams
   ~fastdfe.optimization.Covariate

``BaseInference``
~~~~~~~~~~~~~~~~~

.. autoclass:: fastdfe.base_inference.BaseInference

``JointInference``
~~~~~~~~~~~~~~~~~~

.. autoclass:: fastdfe.joint_inference.JointInference

``Inference``
~~~~~~~~~~~~~

.. autoclass:: fastdfe.abstract_inference.Inference

``InferenceResult``
~~~~~~~~~~~~~~~~~~~

.. autoclass:: fastdfe.base_inference.InferenceResult

``SharedParams``
~~~~~~~~~~~~~~~~

.. autoclass:: fastdfe.optimization.SharedParams

``Covariate``
~~~~~~~~~~~~~

.. autoclass:: fastdfe.optimization.Covariate
