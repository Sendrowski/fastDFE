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

.. autoclass:: fastdfe.base_inference.BaseInference

.. autoclass:: fastdfe.joint_inference.JointInference

.. autoclass:: fastdfe.abstract_inference.Inference

.. autoclass:: fastdfe.base_inference.InferenceResult

.. autoclass:: fastdfe.optimization.SharedParams

.. autoclass:: fastdfe.optimization.Covariate
