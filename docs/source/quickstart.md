# Quickstart
The easiest way to get started is by using the {class}`~fastdfe.base_inference.BaseInference` class, which infers the DFE from a single pair of frequency spectra, one `neutral` and one `selected`. In this example we create {class}`~sfsutils.spectrum.Spectrum` objects holding the SFS counts and pass them to {class}`~fastdfe.base_inference.BaseInference`. The number of monomorphic sites must be specified: the first and last entries of the counts are the numbers of sites where the ancestral and derived allele is fixed, respectively. By default, only the deleterious part of the DFE is inferred (cf. {attr}`~fastdfe.base_inference.BaseInference.fixed_params`).

```{code-cell} python
:tags: [remove-cell]
import matplotlib

matplotlib.rcParams['figure.figsize'] = [4.8, 3.3]
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.8, repr.plot.height = 3.3)
```

```{code-cell} python
import fastdfe as fd

inf = fd.BaseInference(
    sfs_neut=fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 0]),
    sfs_sel=fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 0]),
    do_bootstrap=False
)

inf.run();
```

```{code-cell} python
:tags: [remove-cell]
assert inf.params_mle['S_d'] < 0 and 0 < inf.params_mle['b'] < 1
assert inf.runs['all.S_d'].std() < 0.01 * abs(inf.params_mle['S_d'])
```

```{code-cell} r
library(fastdfe)
fd <- load_fastdfe()

inf <- fd$BaseInference(
  sfs_neut = fd$Spectrum(c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 0)),
  sfs_sel = fd$Spectrum(c(797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 0)),
  do_bootstrap = FALSE
)

sfs_modelled <- inf$run()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(inf$params_mle$S_d < 0, inf$params_mle$b > 0, inf$params_mle$b < 1)
stopifnot(sd(inf$runs[["all.S_d"]]) < 0.01 * abs(inf$params_mle$S_d))
```

+++
``fastdfe`` uses maximum likelihood estimation (MLE) to find the DFE. By default, 10 local optimization runs are carried out to make sure a reasonably good global optimum has been found. The DFE furthermore needs to be parametrized, where {class}`~fastdfe.parametrization.GammaExpParametrization` is used by default. The standard deviation across optimization runs is also reported to give an idea of the reliability of the estimates. In this case, the standard deviations are low, indicating that the estimates are stable.

+++
We can now plot the inferred DFE in discretized form (cf. {func}`~fastdfe.base_inference.BaseInference.plot_discretized`).

```{code-cell} python
inf.plot_discretized();
```

```{code-cell} r
p <- inf$plot_discretized()
```

+++
We can also plot a comparison of the `selected` modelled and observed SFS (cf. {func}`~fastdfe.base_inference.BaseInference.plot_sfs_comparison`).

```{code-cell} python
inf.plot_sfs_comparison();
```

```{code-cell} r
p <- inf$plot_sfs_comparison()
```

+++
## Bootstrapping

To quantify uncertainty, we can perform parametric bootstrapping (cf. {func}`~fastdfe.base_inference.BaseInference.bootstrap`).

```{code-cell} python
inf.bootstrap(n_samples=100)

inf.plot_discretized();
```

```{code-cell} python
:tags: [remove-cell]
assert len(inf.bootstraps) == 100 and inf.bootstraps['likelihoods_std'].mean() < 0.5
```

```{code-cell} r
bootstraps <- inf$bootstrap(n_samples = 100L)

p <- inf$plot_discretized()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(nrow(inf$bootstraps) == 100, mean(inf$bootstraps$likelihoods_std) < 0.5)
```

+++
By default, 2 optimization runs are performed per bootstrap sample, taking the best result (cf. {attr}`~fastdfe.base_inference.BaseInference.n_bootstrap_retries`). The standard deviation across runs is computed for each bootstrap sample, and the average of these standard deviations across all samples is reported to summarize the uncertainty of the bootstrap estimates. In this case, the uncertainty is low, indicating reliable bootstrap estimates.
