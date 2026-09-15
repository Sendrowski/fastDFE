# DFE inference
## Estimating the deleterious DFE
A short overview of basic DFE inference and bootstrapping is available in the {doc}`quickstart guide <quickstart>`. DFE inference requires one `neutral` and one `selected` SFS. In this example we use the bundled `Betula pendula` (silver birch) data. By default, bootstrapping is performed automatically, and the inference estimates only the deleterious component of the DFE using {class}`~fastdfe.parametrization.GammaExpParametrization`.

```{code-cell} python
:tags: [remove-cell]
import matplotlib
import numpy as np

matplotlib.rcParams['figure.figsize'] = [4.8, 3.3]
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9

np.set_printoptions(legacy='1.21')

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

sfs_neut = fd.Spectrum([177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 0])
sfs_sel = fd.Spectrum([797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 0])

inf = fd.BaseInference(
    sfs_neut=sfs_neut,
    sfs_sel=sfs_sel
)

inf.run()

inf.plot_discretized();
```

```{code-cell} r
library(fastdfe)
fd <- load_fastdfe()

sfs_neut <- fd$Spectrum(c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 0))
sfs_sel <- fd$Spectrum(c(797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 0))

inf <- fd$BaseInference(
  sfs_neut = sfs_neut,
  sfs_sel = sfs_sel
)

sfs_modelled <- inf$run()

p <- inf$plot_discretized()
```

+++
It is good practice to check the variability of estimates across optimization runs to ensure stability. Here, both the standard deviations across initial runs and across runs within each bootstrap sample are low, indicating stable estimates. Individual runs and bootstrap results can be inspected in the corresponding dataframes (cf. {attr}`~fastdfe.base_inference.BaseInference.runs` and {attr}`~fastdfe.base_inference.BaseInference.bootstraps`).

```{code-cell} python
inf.runs.select_dtypes('number')
```

```{code-cell} python
:tags: [remove-cell]
assert inf.runs['all.S_d'].std() < 0.01 * abs(inf.params_mle['S_d'])
```

```{code-cell} python
inf.bootstraps.select_dtypes('number').head(10)
```

```{code-cell} r
inf$runs[sapply(inf$runs, is.numeric)]
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(sd(inf$runs[["all.S_d"]]) < 0.01 * abs(inf$params_mle$S_d))
```

```{code-cell} r
head(inf$bootstraps[sapply(inf$bootstraps, is.numeric)], 10)
```

+++
We can also plot the parameter distributions across bootstrap samples to visualize uncertainty. The mean strength of deleterious selection `S_d` often reaches the lower bound of `-1e5`. A different DFE parametrization or a more complex DFE model might be more appropriate here. The spectra used in this example are also far from exemplary, as they contain few SNPs and have a small sample size.

```{code-cell} python
:tags: [full-width]
inf.bootstraps[['S_d', 'b']].hist(figsize=(8.5, 2.5), grid=False, xrot=30);
```

```{code-cell} python
:tags: [remove-cell]
assert (inf.bootstraps.S_d <= -0.9999e5).mean() > 0.1
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 8.5, repr.plot.height = 2.5)
```

```{code-cell} r
:tags: [full-width]
par(mfrow = c(1, 2), mar = c(2.5, 3, 2, 1))
hist(inf$bootstraps$S_d, main = "S_d", xlab = "", col = "#1f77b4", border = "white")
hist(inf$bootstraps$b, main = "b", xlab = "", col = "#1f77b4", border = "white")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(mean(inf$bootstraps$S_d <= -0.9999e5) > 0.1)
options(repr.plot.width = 4.8, repr.plot.height = 3.3)
```

+++
We can also inspect how parameters covary.

```{code-cell} python
inf.bootstraps.assign(S_d=inf.bootstraps.S_d.abs()).plot.scatter('S_d', 'b', logx=True);
```

```{code-cell} python
:tags: [remove-cell]
assert np.corrcoef(np.log(inf.bootstraps.S_d.abs()), inf.bootstraps.b)[0, 1] < -0.5
```

```{code-cell} r
par(mar = c(4, 4, 1, 1))
plot(abs(inf$bootstraps$S_d), inf$bootstraps$b, log = "x", xlab = "S_d", ylab = "b", pch = 16, col = "#1f77b4")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(cor(log(abs(inf$bootstraps$S_d)), inf$bootstraps$b) < -0.5)
```

+++
We observe a strong dependence between the mean `S_d` and the shape parameter `b` of {class}`~fastdfe.parametrization.GammaExpParametrization`. This is because a large fraction of moderately deleterious mutations and a smaller fraction of strongly deleterious mutations can leave a similar signal in the SFS. Spectra with larger sample sizes might facilitate disentangling the two.

+++
## Estimating beneficial effects
Parameters can be held fixed during maximum-likelihood optimization, and this was already done internally in the example above. By default, ``fastdfe`` infers only the deleterious DFE, fixes the ancestral-allele misidentification rate `eps`, and assumes semi-dominant mutations (`h = 0.5`) (see {attr}`~fastdfe.base_inference.BaseInference.fixed_params`). Here, we estimate the full DFE, allowing for beneficial mutations by letting the parameters `S_b` and `p_b` of {class}`~fastdfe.parametrization.GammaExpParametrization` vary, while `eps` and `h` remain fixed. The fixed parameters are grouped under the key `all`, meaning these settings apply to all SFS types, which matters when running joint inference (cf. {class}`~fastdfe.joint_inference.JointInference`).

```{code-cell} python
inf = fd.BaseInference(
    sfs_neut=sfs_neut,
    sfs_sel=sfs_sel,
    fixed_params=dict(all=dict(eps=0, h=0.5))
)

inf.run()

inf.plot_discretized();
```

```{code-cell} r
inf <- fd$BaseInference(
  sfs_neut = sfs_neut,
  sfs_sel = sfs_sel,
  fixed_params = list(all = list(eps = 0, h = 0.5))
)

sfs_modelled <- inf$run()

p <- inf$plot_discretized()
```

+++
The inferred full DFE shows substantial uncertainty, which is expected with a small sample and few SNPs. This is most pronounced for the [-1, 0] and [0, 1] bins, in which mutations are effectively neutral and provide little signal. Adjusting the discretization intervals can help reveal the structure more clearly (cf. {func}`~fastdfe.base_inference.BaseInference.plot_discretized`).

```{code-cell} python
inf.plot_discretized(intervals=[-np.inf, -100, -10, -1, 1, np.inf]);
```

```{code-cell} r
p <- inf$plot_discretized(intervals = c(-Inf, -100, -10, -1, 1, Inf))
```

+++
## Divergence counts
Besides polymorphism, ``fastdfe`` can incorporate divergence counts, the number of fixed differences (substitutions) to an outgroup, into the likelihood, much like `polydfe`. The last entry of an SFS is the fixed-derived (divergence) class. To make use of divergence, the divergence target sizes `n_sites_div_neut` and `n_sites_div_sel` are additionally passed to the inference. They are the numbers of mutational target sites over which divergence was counted, which may differ from the polymorphism target size.

When both spectra carry a separate divergence target size, divergence is included in the likelihood automatically (see {attr}`~fastdfe.base_inference.BaseInference.include_divergence`). The selected divergence then helps constrain the beneficial part of the DFE, and {math}`\alpha`, the proportion of beneficial substitutions, can be estimated McDonald–Kreitman style from the observed divergence.

```{code-cell} python
# the SFS runs from the monomorphic (ancestral) class through the polymorphic bins to the
# fixed-derived (divergence) class in the last entry
sfs_neut = fd.Spectrum([171150, 997, 441, 228, 156, 117, 114, 83, 105, 109, 6500])
sfs_sel = fd.Spectrum([793221, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 14000])

# specifying the divergence target sizes includes divergence in the likelihood
inf = fd.BaseInference(
    sfs_neut=sfs_neut,
    sfs_sel=sfs_sel,
    n_sites_div_neut=180000,
    n_sites_div_sel=810000,
    fixed_params=dict(all=dict(eps=0, h=0.5))
)

inf.run()

inf.plot_discretized();
```

```{code-cell} r
# the SFS runs from the monomorphic (ancestral) class through the polymorphic bins to the
# fixed-derived (divergence) class in the last entry
sfs_neut <- fd$Spectrum(c(171150, 997, 441, 228, 156, 117, 114, 83, 105, 109, 6500))
sfs_sel <- fd$Spectrum(c(793221, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 14000))

# specifying the divergence target sizes includes divergence in the likelihood
inf <- fd$BaseInference(
  sfs_neut = sfs_neut,
  sfs_sel = sfs_sel,
  n_sites_div_neut = 180000,
  n_sites_div_sel = 810000,
  fixed_params = list(all = list(eps = 0, h = 0.5))
)

sfs_modelled <- inf$run()

p <- inf$plot_discretized()
```

+++
The estimator used by {func}`~fastdfe.base_inference.BaseInference.get_alpha` follows the inference mode, but can be switched explicitly via its `use_divergence` argument, which allows {math}`\alpha` estimated from divergence to be compared with {math}`\alpha` estimated from polymorphism alone. {func}`~fastdfe.base_inference.BaseInference.get_omega` returns {math}`\omega`, the ratio of non-synonymous to synonymous substitution rates ({math}`d_N/d_S`), and {func}`~fastdfe.base_inference.BaseInference.get_omega_a` returns its adaptive component {math}`\omega_a = \alpha\,\omega`.

```{code-cell} python
print(f'alpha (with divergence):   {inf.get_alpha():.7g}')
print(f'alpha (polymorphism only): {inf.get_alpha(use_divergence=False):.7g}')
print(f'omega:   {inf.get_omega():.7g}')
print(f'omega_a: {inf.get_omega_a():.7g}')
```

```{code-cell} python
:tags: [remove-cell]
assert 0 < inf.get_alpha() < 1 and 0 < inf.get_alpha(use_divergence=False) < 1
```

```{code-cell} r
cat('alpha (with divergence):  ', inf$get_alpha(), '\n')
cat('alpha (polymorphism only):', inf$get_alpha(use_divergence = FALSE), '\n')
cat('omega:  ', inf$get_omega(), '\n')
cat('omega_a:', inf$get_omega_a(), '\n')
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(inf$get_alpha() > 0, inf$get_alpha() < 1)
```

+++
## Ancestral-allele misidentification
We can also adjust for ancestral-allele misidentification by letting parameter `eps` vary. `eps` is the probability that an allele is misidentified as derived when it is actually ancestral, and vice versa (cf. {meth}`~sfsutils.spectrum.Spectrum.misidentify`). This can correct biases to the SFS caused by mis-polarization, but `eps` is somewhat difficult to interpret because it is applied simultaneously to both the neutral and selected SFS. In addition, `eps` assumes the fraction of ancestral misidentification to be constant across site classes, whereas in practice errors may differ across classes. Nevertheless, below, we infer the full DFE while allowing `eps` to vary.

```{code-cell} python
inf = fd.BaseInference(
    sfs_neut=sfs_neut,
    sfs_sel=sfs_sel,
    fixed_params=dict(all=dict(h=0.5))
)

inf.run();
```

```{code-cell} r
inf <- fd$BaseInference(
  sfs_neut = sfs_neut,
  sfs_sel = sfs_sel,
  fixed_params = list(all = list(h = 0.5))
)

sfs_modelled <- inf$run()
```

+++
The bootstrap distribution of `eps`:

```{code-cell} python
inf.bootstraps.eps.hist(grid=False, figsize=(4.8, 2.5));
```

```{code-cell} python
:tags: [remove-cell]
assert inf.params_mle['eps'] < 0.05
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.8, repr.plot.height = 2.5)
```

```{code-cell} r
par(mar = c(2.5, 3, 1, 1))
hist(inf$bootstraps$eps, main = "", xlab = "", col = "#1f77b4", border = "white")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(inf$params_mle$eps < 0.05)
options(repr.plot.width = 4.8, repr.plot.height = 3.3)
```

+++
`eps` is estimated to be rather low, indicating that ancestral-allele misidentification is not a major issue in this dataset, or at least that including it does not significantly improve the model fit. We can check this in a more principled way by performing a likelihood-ratio test as done below.

+++
## Nested model comparison
The significance of including ancestral-allele misidentification and beneficial mutations can be assessed with likelihood ratio tests, using {func}`~fastdfe.base_inference.BaseInference.plot_nested_models`. The LRTs compare the likelihood of the inferred DFE to the likelihood of a nested model where some parameters are held fixed. Alternatively, {func}`~fastdfe.base_inference.BaseInference.compare_nested` directly compares two nested models.

```{code-cell} python
# set logging level to warning to avoid cluttering
fd.logger.setLevel('WARNING')

inf.plot_nested_models()

fd.logger.setLevel('INFO')
```

```{code-cell} python
:tags: [remove-cell]
P, _ = inf.compare_nested_models()
assert all(p > 0.05 for p in P.flatten() if p is not None)
```

```{code-cell} r
# set logging level to warning to avoid cluttering
fd$logger$setLevel('WARNING')

p <- inf$plot_nested_models()

fd$logger$setLevel('INFO')
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(all(unlist(inf$compare_nested_models()[[1]]) > 0.05))
```

+++
Including ancestral allele misidentification or beneficial mutations does not significantly improve the fit.

+++
## Dominance effects
By default, ``fastdfe`` assumes semi-dominance (`h = 0.5`), which is more or less appropriate depending on the organism and type of mutations considered. We can change the dominance coefficient to a different value of `h` if we believe this is more appropriate. However, in practice, `h` often depends on the strength of selection, with more deleterious mutations being more recessive. To model this, we can specify a callback function that returns the dominance coefficient as a function of the scaled selection coefficient `S = 4 Ne s`.

In the example below, we use an exponential decay: `h` is about 0.4 for neutral mutations and approaches 0 for strongly deleterious ones. The callback also receives `h` itself, allowing the dominance function to be parametrized and optimized. For simplicity, this parameter is still called `h`. Its bounds can be set via {attr}`~fastdfe.base_inference.BaseInference.bounds`.

```{code-cell} python
inf = fd.BaseInference(
    sfs_neut=sfs_neut,
    sfs_sel=sfs_sel,
    fixed_params=dict(all=dict(eps=0, h=0, p_b=0, S_b=1)),
    h_callback=lambda h, S: 0.4 * np.exp(-0.1 * abs(S))
)

inf.run();
```

```{code-cell} r
inf <- fd$BaseInference(
  sfs_neut = sfs_neut,
  sfs_sel = sfs_sel,
  fixed_params = list(all = list(eps = 0, h = 0, p_b = 0, S_b = 1)),
  h_callback = function(h, S) 0.4 * exp(-0.1 * abs(S))
)

sfs_modelled <- inf$run()
```

+++
We compare the inferred DFE under this dominance relationship to that of the default semi-dominant model.

```{code-cell} python
inf2 = fd.BaseInference(
    sfs_neut=sfs_neut,
    sfs_sel=sfs_sel,
    fixed_params=dict(all=dict(eps=0, h=0.5, p_b=0, S_b=1))
)

inf2.run()

fd.DFE.plot_many([inf.get_dfe(), inf2.get_dfe()], labels=['partly recessive', 'h=0.5']);
```

```{code-cell} python
:tags: [remove-cell]
# the partly recessive model puts more mass on strongly deleterious mutations
assert inf.get_discretized()[0][0] > inf2.get_discretized()[0][0]
```

```{code-cell} r
inf2 <- fd$BaseInference(
  sfs_neut = sfs_neut,
  sfs_sel = sfs_sel,
  fixed_params = list(all = list(eps = 0, h = 0.5, p_b = 0, S_b = 1))
)

sfs_modelled <- inf2$run()

p <- fd$DFE$plot_many(list(inf$get_dfe(), inf2$get_dfe()), labels = c('partly recessive', 'h=0.5'))
```

```{code-cell} r
:tags: [remove-cell]
# the partly recessive model puts more mass on strongly deleterious mutations
stopifnot(inf$get_discretized()[[1]][1] > inf2$get_discretized()[[1]][1])
```

+++
Assuming that mutations are partly recessive leads to a more deleterious inferred DFE, since stronger selection is necessary to remove a similar amount of recessive mutations.

We can also let `h` vary when inferring the DFE (cf. the {doc}`simulation guide <simulation>`).

+++
## Folded inference
To infer the DFE from a folded SFS, folded spectra are passed to {class}`~fastdfe.base_inference.BaseInference`. Folded inference is performed whenever the spectra are folded, i.e., when all entries where the derived allele is the major allele are zero. Folded spectra contain little information on beneficial mutations, so we only infer the deleterious part of the DFE here.

```{code-cell} python
:tags: [full-width]
import matplotlib.pyplot as plt

inf = fd.BaseInference(
    sfs_neut=sfs_neut.fold(),
    sfs_sel=sfs_sel.fold()
)

inf.run()

# plot the inferred DFE and the SFS comparison
_, (ax1, ax2) = plt.subplots(ncols=2, figsize=(7, 3.2))

inf.plot_discretized(ax=ax1, show=False, intervals=[-np.inf, -100, -10, -1, 0])
inf.plot_sfs_comparison(ax=ax2);
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 7, repr.plot.height = 3.2)
```

```{code-cell} r
:tags: [full-width]
inf <- fd$BaseInference(
  sfs_neut = sfs_neut$fold(),
  sfs_sel = sfs_sel$fold()
)

sfs_modelled <- inf$run()

# plot the inferred DFE and the SFS comparison
p1 <- inf$plot_discretized(show = FALSE, intervals = c(-Inf, -100, -10, -1, 0))
p2 <- inf$plot_sfs_comparison(show = FALSE)

cowplot::plot_grid(p1, p2, ncol = 2)
```

```{code-cell} r
:tags: [remove-cell]
options(repr.plot.width = 4.8, repr.plot.height = 3.3)
```

+++
## Serialization
Inference objects can be serialized to JSON files for later use (cf. {func}`~fastdfe.base_inference.BaseInference.to_file`).

```{code-cell} python
# save the inference object to a file, which BaseInference.from_file restores
inf.to_file("serialized.json")

# save a short summary to a file
inf.get_summary().to_file("summary.json")
```

```{code-cell} python
:tags: [remove-cell]
assert fd.BaseInference.from_file("serialized.json").params_mle == inf.params_mle
```

```{code-cell} r
# save the inference object to a file, which BaseInference$from_file restores
inf$to_file("serialized.json")

# save a short summary to a file
inf$get_summary()$to_file("summary.json")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(isTRUE(all.equal(fd$BaseInference$from_file("serialized.json")$params_mle, inf$params_mle)))
```

+++
## Joint inference
``fastdfe`` supports joint inference of several SFS types, where any parameters can be shared between types. In this example, we create a {class}`~fastdfe.joint_inference.JointInference` object with two types that share ``S_d``, the mean selection coefficient of deleterious mutations (cf. {class}`~fastdfe.parametrization.GammaExpParametrization`). For more complex stratifications, see the {class}`~sfsutils.parser.Parser` module.

```{code-cell} python
# neutral SFS for two types
sfs_neut = fd.Spectra(dict(
    pendula=[177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 0],
    pubescens=[172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31]
))

# selected SFS for two types
sfs_sel = fd.Spectra(dict(
    pendula=[797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 0],
    pubescens=[791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41]
))

inf = fd.JointInference(
    sfs_neut=sfs_neut,
    sfs_sel=sfs_sel,
    shared_params=[fd.SharedParams(types=["pendula", "pubescens"], params=["S_d"])]
)

inf.run();
```

```{code-cell} r
# neutral SFS for two types
sfs_neut <- fd$Spectra(list(
  pendula = c(177130, 997, 441, 228, 156, 117, 114, 83, 105, 109, 0),
  pubescens = c(172528, 3612, 1359, 790, 584, 427, 325, 234, 166, 76, 31)
))

# selected SFS for two types
sfs_sel <- fd$Spectra(list(
  pendula = c(797939, 1329, 499, 265, 162, 104, 117, 90, 94, 119, 0),
  pubescens = c(791106, 5326, 1741, 1005, 756, 546, 416, 294, 177, 104, 41)
))

inf <- fd$JointInference(
  sfs_neut = sfs_neut,
  sfs_sel = sfs_sel,
  shared_params = list(fd$SharedParams(types = c("pendula", "pubescens"), params = list("S_d")))
)

sfs_modelled <- inf$run()
```

+++
{class}`~fastdfe.joint_inference.JointInference` runs both the joint inference and the marginal inferences, where each type is inferred separately. To see this better, we plot the inferred parameters for the different inference types.

```{code-cell} python
inf.plot_inferred_parameters();
```

```{code-cell} python
:tags: [remove-cell]
assert inf.params_mle['pendula']['S_d'] == inf.params_mle['pubescens']['S_d']
```

```{code-cell} r
p <- inf$plot_inferred_parameters()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(inf$params_mle$pendula$S_d == inf$params_mle$pubescens$S_d)
```

+++
``marginal.pendula`` and ``marginal.pubescens`` are the marginal inferences for the respective type. ``marginal.all`` is the marginal inference obtained by adding up the spectra of all types. ``joint.pendula`` and ``joint.pubescens`` are the joint inferences for the respective type. We can see that ``S_d`` is indeed shared between the two. The parameter ``alpha`` in the plot denotes the proportion of beneficial non-synonymous substitutions. Each marginal inference is a {class}`~fastdfe.base_inference.BaseInference` object itself and is available through {attr}`~fastdfe.joint_inference.JointInference.marginal_inferences`.

We can now also investigate to what extent the inferred DFEs differ:

```{code-cell} python
inf.plot_discretized();
```

```{code-cell} r
p <- inf$plot_discretized()
```

+++
### Model comparison
We can obtain information about the goodness of fit achieved by sharing the parameter by performing a likelihood ratio test (cf. {func}`~fastdfe.joint_inference.JointInference.perform_lrt_shared`). This compares the likelihood of the joint inference with the product of the marginal likelihoods.

```{code-cell} python
inf.perform_lrt_shared()
```

```{code-cell} python
:tags: [remove-cell]
assert inf.perform_lrt_shared() > 0.05
```

```{code-cell} r
inf$perform_lrt_shared()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(inf$perform_lrt_shared() > 0.05)
```

+++
The test is not significant, indicating that the simpler model of sharing the parameters explains the data sufficiently well. Indeed, the inferred parameters of the joint and the marginal inferences differ little.

+++
## Covariates
{class}`~fastdfe.joint_inference.JointInference` also supports covariates associated with the different SFS types. This provides more powerful model testing and reduces the number of parameters that need to be estimated for the joint inference. For a more interesting example, we stratify the SFS of `B. pendula` by the sites' ancestral base, as described in more detail in the ``sfsutils`` [stratifications reference](https://sfsutils.readthedocs.io/en/latest/modules/stratification.html).

```{code-cell} python
parser = fd.Parser(
    n=10,
    source="https://github.com/Sendrowski/fastDFE/"
           "blob/dev/resources/genome/betula/"
           "all.polarized.deg.subset.200000.vcf.gz?raw=true",
    stratifications=[fd.DegeneracyStratification(), fd.AncestralBaseStratification()]
)

spectra: fd.Spectra = parser.parse()

spectra.plot();
```

```{code-cell} r
parser <- fd$Parser(
  n = 10,
  source = paste0(
    "https://github.com/Sendrowski/fastDFE/",
    "blob/dev/resources/genome/betula/",
    "all.polarized.deg.subset.200000.vcf.gz?raw=true"
  ),
  stratifications = list(fd$DegeneracyStratification(), fd$AncestralBaseStratification())
)

spectra <- parser$parse()

p <- spectra$plot()
```

+++
We now create the inference object from the spectra. In this contrived example we make up some covariates that covary with ``S_d``, the mean strength of negative selection. Covariates introduce a linear relationship by default, but this can be modified by specifying a custom callback function (see {class}`~fastdfe.optimization.Covariate`).

```{code-cell} python
inf = fd.JointInference(
    sfs_neut=spectra[['neutral.*']].merge_groups(1),
    sfs_sel=spectra[['selected.*']].merge_groups(1),
    covariates=[fd.Covariate(param='S_d', values=dict(A=1, C=2, T=3, G=4))],
    n_runs=50  # increase number of initial runs for stability
)

inf.run();
```

```{code-cell} r
inf <- fd$JointInference(
  sfs_neut = spectra$select('neutral.*')$merge_groups(1L),
  sfs_sel = spectra$select('selected.*')$merge_groups(1L),
  covariates = list(fd$Covariate(param = 'S_d', values = list(A = 1, C = 2, T = 3, G = 4))),
  n_runs = 50L  # increase number of initial runs for stability
)

sfs_modelled <- inf$run()
```

+++
The inferred parameters:

```{code-cell} python
inf.plot_inferred_parameters();
```

```{code-cell} r
p <- inf$plot_inferred_parameters()
```

+++
``S_d`` shows little variation across the jointly inferred types, because it does not change linearly with respect to the arbitrary covariates specified. Indeed, the median of the covariate coefficient across all bootstrap replicates is close to zero. Covariates are named ``c0``, ``c1``, etc., by default.

```{code-cell} python
inf.bootstraps['A.c0'].median()
```

```{code-cell} python
:tags: [remove-cell]
assert abs(inf.bootstraps['A.c0'].median()) < 0.1
```

```{code-cell} r
median(inf$bootstraps[['A.c0']])
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(median(inf$bootstraps[["A.c0"]])) < 0.1)
```

+++
### Model comparison
We can perform a likelihood ratio test to see whether including the covariates produces a significantly better fit than simply sharing the parameter in question among the types (cf. {func}`~fastdfe.joint_inference.JointInference.perform_lrt_covariates`).

```{code-cell} python
inf.perform_lrt_covariates()
```

```{code-cell} python
:tags: [remove-cell]
assert inf.perform_lrt_covariates() > 0.05
```

```{code-cell} r
inf$perform_lrt_covariates()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(inf$perform_lrt_covariates() > 0.05)
```

+++
As expected, the specified covariates do not improve the fit significantly.
