# DFE parametrizations

The DFE needs to be parametrized in some way in order to be amenable to maximum likelihood estimation. {class}`~fastdfe.parametrization.GammaExpParametrization` is used by default. Other parametrizations are also implemented (cf. {mod}`~fastdfe.parametrization`), and custom parametrizations can be created by subclassing {class}`~fastdfe.parametrization.Parametrization`.

To see how the parametrization affects the shape of the DFE, we use our example data for `B. pendula`.

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = [4.8, 3.3]
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9

import fastdfe as fd

fd.Settings.parallelize = False
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.8, repr.plot.height = 3.3)

settings <- reticulate::import("fastdfe")$Settings
settings$parallelize <- FALSE
```

```{code-cell} python
import fastdfe as fd

# only show very important log messages
fd.logger.setLevel('FATAL')

parametrizations = [
    fd.GammaExpParametrization(),
    fd.DiscreteFractionalParametrization(),
    fd.GammaDiscreteParametrization(),
    fd.DisplacedGammaParametrization()
]

inferences = []
for model in parametrizations:
    inf = fd.BaseInference(
        sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 0]),
        sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 0]),
        fixed_params=dict(all=dict(h=0.5, eps=0)),
        model=model
    )

    inf.run()

    inferences.append(inf)
```

```{code-cell} r
library(fastdfe)
fd <- load_fastdfe()

# only show very important log messages
fd$logger$setLevel('FATAL')

parametrizations <- list(
  fd$GammaExpParametrization(),
  fd$DiscreteFractionalParametrization(),
  fd$GammaDiscreteParametrization(),
  fd$DisplacedGammaParametrization()
)

inferences <- list()
for (model in parametrizations) {
  inf <- fd$BaseInference(
    sfs_neut = fd$Spectrum(c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 0)),
    sfs_sel = fd$Spectrum(c(797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 0)),
    fixed_params = list(all = list(h = 0.5, eps = 0)),
    model = model
  )

  sfs_modelled <- inf$run()

  inferences <- c(inferences, inf)
}
```

+++
The inferred DFEs are plotted in discretized form.

```{code-cell} python
import numpy as np

fd.Inference.plot_discretized(
    inferences=inferences,
    labels=['GammaExp', 'DiscreteFractional', 'GammaDiscrete', 'DisplacedGamma'],
    intervals=[-np.inf, -100, -10, -1, 1, np.inf]
);
```

```{code-cell} python
:tags: [remove-cell]
# the discrete fractional parametrization has the widest confidence intervals
widths = [inf.get_discretized(intervals=np.array([-np.inf, -100, -10, -1, 1, np.inf]))[1].sum() for inf in inferences]
assert np.argmax(widths) == 1
```

```{code-cell} r
p <- fd$Inference$plot_discretized(
  inferences = inferences,
  labels = c('GammaExp', 'DiscreteFractional', 'GammaDiscrete', 'DisplacedGamma'),
  intervals = c(-Inf, -100, -10, -1, 1, Inf)
)
```

```{code-cell} r
:tags: [remove-cell]
# the discrete fractional parametrization has the widest confidence intervals
widths <- sapply(inferences, function(inf) sum(inf$get_discretized(intervals = reticulate::np_array(c(-Inf, -100, -10, -1, 1, Inf)))[[2]]))
stopifnot(which.max(widths) == 2)
```

+++
The overall shape is similar, but {class}`~fastdfe.parametrization.DiscreteFractionalParametrization` shows noticeably wider confidence intervals. In general, estimating the full DFE with a sample size of 10 and the limited SNP count used here leads to substantial uncertainty.
