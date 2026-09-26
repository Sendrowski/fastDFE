# SFS simulation
``fastdfe`` provides a {class}`~fastdfe.simulation.Simulation` module to generate an expected SFS under a specific DFE, and potentially demographic nuisance parameters. In fact, when inferring the DFE with ``fastdfe``, the expected SFS is repeatedly simulated for different parameters to find the maximum likelihood estimates. Below, we illustrate how to use the simulation module to generate SFS data under a given DFE. To do this, we need to specify a neutral SFS ({attr}`~fastdfe.simulation.Simulation.sfs_neut`), which is informative on the population sample size, population mutation rate, the number of sites, and possibly demography. {func}`~fastdfe.simulation.Simulation.get_neutral_sfs` can be used to obtain a neutral SFS under a constant panmictic population. We also specify the DFE parameters and parametrization. By default, no ancestral misidentification is assumed (`eps=0`), and mutations are assumed to be semi-dominant (`h=0.5`).

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

# create simulation object by specifying neutral SFS and DFE
sim = fd.Simulation(
    sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
    params=dict(S_d=-300, b=0.3, p_b=0, S_b=1),
    model=fd.GammaExpParametrization()
)
```

```{code-cell} r
library(fastdfe)
fd <- load_fastdfe()

# create simulation object by specifying neutral SFS and DFE
sim <- fd$Simulation(
  sfs_neut = fd$Simulation$get_neutral_sfs(n = 20L, n_sites = 1e8, theta = 1e-4),
  params = list(S_d = -300, b = 0.3, p_b = 0, S_b = 1),
  model = fd$GammaExpParametrization()
)
```

+++
We plot the DFE used for the simulation, which is purely deleterious and follows a gamma distribution.

```{code-cell} python
sim.dfe.plot();
```

```{code-cell} r
p <- sim$dfe$plot()
```

+++
We now run the simulation and plot the expected SFS for both neutral (specified) and selected (simulated) sites.

```{code-cell} python
sim.run()

sim.get_spectra().plot();
```

```{code-cell} r
sfs_sel <- sim$run()

p <- sim$get_spectra()$plot()
```

+++
We can now use the simulated SFS to infer the DFE parameters and assess how closely the inference recovers the true DFE used to generate the data.

```{code-cell} python
inf = fd.BaseInference(
    sfs_neut=sim.sfs_neut,
    sfs_sel=sim.sfs_sel,
    fixed_params=dict(all=dict(eps=0, h=0.5, p_b=0, S_b=1))
)

inf.run()

fd.DFE.plot_many([sim.dfe, inf.get_dfe()], labels=['True DFE', 'Inferred DFE']);
```

```{code-cell} python
:tags: [remove-cell]
assert abs(inf.params_mle['S_d'] / -300 - 1) < 0.01 and abs(inf.params_mle['b'] / 0.3 - 1) < 0.01
```

```{code-cell} r
inf <- fd$BaseInference(
  sfs_neut = sim$sfs_neut,
  sfs_sel = sim$sfs_sel,
  fixed_params = list(all = list(eps = 0, h = 0.5, p_b = 0, S_b = 1))
)

sfs_modelled <- inf$run()

p <- fd$DFE$plot_many(list(sim$dfe, inf$get_dfe()), labels = c('True DFE', 'Inferred DFE'))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(inf$params_mle$S_d / -300 - 1) < 0.01, abs(inf$params_mle$b / 0.3 - 1) < 0.01)
```

+++
The DFE is recovered closely, which is expected given the simplicity of the DFE parametrization and the lack of demographic complications. The accuracy of ``fastdfe`` under more realistic conditions, including strong demographic distortions, population substructure, background selection, dominance and small sample sizes, is assessed with ``SLiM`` forward simulations in Appendix C of {cite}`primatedfe`.

+++
We may want to see how the expected SFS changes as we vary the degree of dominance. Here, we run a second simulation with the same parameters except that we set `h=0.3` so that mutations are partially recessive.

```{code-cell} python
sim2 = fd.Simulation(
    sfs_neut=fd.Simulation.get_neutral_sfs(n=20, n_sites=1e8, theta=1e-4),
    params=dict(S_d=-300, b=0.3, p_b=0, S_b=1, h=0.3),
    model=fd.GammaExpParametrization()
)

sim2.run()

fd.Spectra.from_spectra(dict(
    neutral=sim.sfs_neut,
    semidominant=sim.sfs_sel,
    recessive=sim2.sfs_sel
)).plot();
```

```{code-cell} python
:tags: [remove-cell]
assert sim2.sfs_sel.data[1] > sim.sfs_sel.data[1]
```

```{code-cell} r
sim2 <- fd$Simulation(
  sfs_neut = fd$Simulation$get_neutral_sfs(n = 20L, n_sites = 1e8, theta = 1e-4),
  params = list(S_d = -300, b = 0.3, p_b = 0, S_b = 1, h = 0.3),
  model = fd$GammaExpParametrization()
)

sfs_sel <- sim2$run()

p <- fd$Spectra$from_spectra(list(
  neutral = sim$sfs_neut,
  semidominant = sim$sfs_sel,
  recessive = sim2$sfs_sel
))$plot()
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(sim2$sfs_sel$data[2] > sim$sfs_sel$data[2])
```

+++
As expected, partially recessive mutations lead to an excess of rare variants because they are masked in heterozygotes and experience weaker purifying selection. At higher frequencies, the difference from the semidominant case diminishes as derived-allele homozygotes become more common.

+++
We can now infer the DFE under the incorrect assumption that mutations are semidominant (`h=0.5`) and see how this affects the inference result.

```{code-cell} python
inf = fd.BaseInference(
    sfs_neut=sim2.sfs_neut,
    sfs_sel=sim2.sfs_sel
)

inf.run()

fd.DFE.plot_many([sim2.dfe, inf.get_dfe()], labels=['True DFE (h=0.3)', 'Inferred DFE (h=0.5)']);
```

```{code-cell} python
:tags: [remove-cell]
assert inf.params_mle['S_d'] > -300
```

```{code-cell} r
inf <- fd$BaseInference(
  sfs_neut = sim2$sfs_neut,
  sfs_sel = sim2$sfs_sel
)

sfs_modelled <- inf$run()

p <- fd$DFE$plot_many(list(sim2$dfe, inf$get_dfe()), labels = c('True DFE (h=0.3)', 'Inferred DFE (h=0.5)'))
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(inf$params_mle$S_d > -300)
```

+++
Assuming semidominance when mutations are actually partially recessive leads to an inferred DFE shifted toward weaker selection coefficients. This happens because rare variants generated by recessive mutations are misattributed to weaker selection when the dominance reduction in purifying selection is not accounted for.

+++
We can also infer the DFE while letting the dominance coefficient `h` vary during the optimization. To optimize `h`, it needs to be discretized. In this example we use the default grid of 21 points between 0 and 1, with intermediate values obtained by linear interpolation. All required values are precomputed before the optimization, but allowing `h` to vary increases the computational cost because the expected SFS must be simulated for each `h` in the grid. To speed up the precomputation step, we instruct {class}`~fastdfe.base_inference.BaseInference` to use a coarser DFE discretization grid than the default, which should be sufficient for most inference scenarios.

```{code-cell} python
inf = fd.BaseInference(
    sfs_neut=sim2.sfs_neut,
    sfs_sel=sim2.sfs_sel,
    intervals_h=(0, 1, 21),
    intervals_del=(-1.0e+8, -1.0e-5, 100),
    intervals_ben=(1.0e-5, 1.0e4, 100),
    fixed_params=dict(all=dict(eps=0, p_b=0, S_b=1))
)

inf.run()

fd.DFE.plot_many([sim2.dfe, inf.get_dfe()], labels=['True DFE (h=0.3)', 'Inferred DFE']);
```

```{code-cell} r
inf <- fd$BaseInference(
  sfs_neut = sim2$sfs_neut,
  sfs_sel = sim2$sfs_sel,
  intervals_h = c(0, 1, 21),
  intervals_del = c(-1.0e+8, -1.0e-5, 100),
  intervals_ben = c(1.0e-5, 1.0e4, 100),
  fixed_params = list(all = list(eps = 0, p_b = 0, S_b = 1))
)

sfs_modelled <- inf$run()

p <- fd$DFE$plot_many(list(sim2$dfe, inf$get_dfe()), labels = c('True DFE (h=0.3)', 'Inferred DFE'))
```

+++
Allowing `h` to vary during inference recovers the true DFE used for the simulation, but with substantially greater uncertainty. We can examine the bootstrap results to assess how well `h` was estimated.

```{code-cell} python
inf.bootstraps.h.hist(grid=False);
```

```{code-cell} python
:tags: [remove-cell]
assert abs(inf.params_mle['h'] - 0.3) < 0.1
assert inf.bootstraps.h.quantile(0.05) < 0.3 < inf.bootstraps.h.quantile(0.95)
```

```{code-cell} r
par(mar = c(4, 4, 1, 1))
hist(inf$bootstraps$h, main = "", xlab = "h", col = "#1f77b4", border = "white")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(abs(inf$params_mle$h - 0.3) < 0.1)
stopifnot(quantile(inf$bootstraps$h, 0.05) < 0.3, quantile(inf$bootstraps$h, 0.95) > 0.3)
```

+++
`h` is estimated with considerable uncertainty, but the bootstrap distribution still puts substantial mass near the true value `h = 0.3`. How well dominance can be inferred from SFS data alone remains an open question, especially when the DFE is more complex or additional nuisance parameters such as demography are included.

```{code-cell} python
inf.bootstraps.assign(S_d=inf.bootstraps.S_d.abs()).plot.scatter('S_d', 'h', logx=True);
```

```{code-cell} python
:tags: [remove-cell]
import numpy as np

assert np.corrcoef(np.log(inf.bootstraps.S_d.abs()), inf.bootstraps.h)[0, 1] < 0
```

```{code-cell} r
par(mar = c(4, 4, 1, 1))
plot(abs(inf$bootstraps$S_d), inf$bootstraps$h, log = "x", xlab = "S_d", ylab = "h", pch = 16, col = "#1f77b4")
```

```{code-cell} r
:tags: [remove-cell]
stopifnot(cor(log(abs(inf$bootstraps$S_d)), inf$bootstraps$h) < 0)
```

+++
`h` covaries with `S_d`, with larger `h` associated with less strongly deleterious mutations.
