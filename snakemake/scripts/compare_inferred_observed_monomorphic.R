library(fastdfe)

fd <- load_fastdfe()

# url for fastdfe repo
url = "https://github.com/Sendrowski/fastDFE/blob/dev/resources/genome/betula/"

# create parser where we consider monomrphic sites
p <- fd$Parser(
  vcf = paste0(url, "all.with_outgroups.subset.200000.vcf.gz?raw=true"),
  fasta = paste0(url, "genome.subset.1000.fasta.gz?raw=true"),
  gff = paste0(url, "genome.gff.gz?raw=true"),
  target_site_counter = NULL,
  n = 10,
  annotations = c(fd$DegeneracyAnnotation()),
  filtrations = c(fd$CodingSequenceFiltration()),
  stratifications = c(fd$DegeneracyStratification())
)

sfs <- p$parse()

sfs$plot(title = "observed")

# create parser where we infer monomrphic sites
p2 = fd$Parser(
  vcf = paste0(url, "all.with_outgroups.subset.200000.vcf.gz?raw=true"),
  fasta = paste0(url, "genome.subset.1000.fasta.gz?raw=true"),
  gff = paste0(url, "genome.gff.gz?raw=true"),
  target_site_counter = fd$TargetSiteCounter(
    n_samples = 1000000,
    n_target_sites = sum(sfs$n_sites)
  ),
  n = 10,
  annotations = c(fd$DegeneracyAnnotation()),
  filtrations = c(fd$CodingSequenceFiltration()),
  stratifications = c(fd$DegeneracyStratification())
)

sfs2 <- p2$parse()

sfs$plot(title = "inferred")

infs <- list()
spectra <- list(sfs, sfs2)

for (i in seq_along(spectra)) {
  inf <- fd$BaseInference(
    sfs_neut = spectra[[i]]$select('neutral'),
    sfs_sel = spectra[[i]]$select('selected'),
    do_bootstrap = TRUE,
    model = fd$DiscreteFractionalParametrization(),
  )

  inf$run()
  
  infs[[i]] <- inf
}
 
# plot inferred DFEs 
fd$Inference$plot_discretized(infs, labels = c('monomorphic', 'inferred'))
