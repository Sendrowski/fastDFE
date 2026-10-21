"""
Smoke tests replaying the code published in the fastDFE manuscript and the book chapter.

Each test reruns a published listing or a script of the book chapter's workflow (``fastDFEBookChapter``) with
small run, bootstrap and site counts. The manuscript listings read the files their URLs point to from ``resources/``,
and the book chapter's inference scripts read the spectra it produced, copied to ``resources/SFS/book_chapter``.
Its VCF pipeline runs on the betula subset in place of the HGDP data, with the two betula outgroups standing in
for the three great apes, and its parsing scripts read a betula subset annotated beforehand. The tests check that the code runs, not the values it returns.
"""
import shutil
import tempfile
from typing import List, Union

import cyvcf2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import fastdfe as fd
from fastdfe.io_handlers import DummyVariant
from testing import TestCase

BETULA = "resources/genome/betula/"
CHAPTER = "resources/SFS/book_chapter/"
OUTGROUPS = ['ERR2103730', 'ERR2103731']

#: Betula subset with ancestral alleles and degeneracy annotated, read by the parsing scripts
POLARIZED = BETULA + "all.polarized.deg.subset.10000.vcf.gz"


class RecombinationIntensityAnnotation(fd.Annotation):
    """
    Recombination intensity annotation using a genetic map, as defined in the book chapter's ``utils.py``.
    """

    def __init__(self, map_file: str, bins_file: str = None, n_bins: int = 10):
        """
        Create a new annotation instance.
        """
        super().__init__()

        self.bins_file = bins_file

        self.n_bins = n_bins

        self.map: pd.DataFrame = pd.read_csv(map_file, comment='#', sep='\t')

        self.chr_cache = {chr_: df for chr_, df in self.map.groupby("Chr")}

        self.values: List[float] = []

        self.current: pd.Series = self.map.iloc[0]

    def _setup(self, handler: fd.io_handlers.MultiHandler):
        """
        Provide context to the annotator.
        """
        super()._setup(handler)

        handler._reader.add_info_to_header({
            'ID': 'Recombination_intensity',
            'Number': '.',
            'Type': 'Float',
            'Description': 'Recombination intensity in cM/Mb.'
        })

    def _teardown(self):
        """
        Finalize the annotation. Called after all sites have been annotated.
        """
        super()._teardown()

        if self.bins_file is not None:
            pd.DataFrame(np.quantile(self.values, np.linspace(0, 1, self.n_bins + 1))).to_csv(self.bins_file)

    def annotate_site(self, variant: Union['cyvcf2.Variant', DummyVariant]):
        """
        Annotate site.
        """
        if (self.current is not None and variant.CHROM == self.current.Chr and
                self.current.Begin <= variant.POS < self.current.End):
            variant.INFO['Recombination_intensity'] = float(self.current.cMperMb)
            self.values.append(self.current.cMperMb)
            self.n_annotated += 1
            return

        chr_map = self.chr_cache.get(variant.CHROM)
        if chr_map is not None:
            mask = (chr_map.Begin <= variant.POS) & (variant.POS < chr_map.End)
            if mask.any():
                self.current = chr_map[mask].iloc[0]
                self.annotate_site(variant)


class RecombinationIntensityStratification(fd.Stratification):
    """
    Recombination intensity stratification, as defined in the book chapter's ``utils.py``.
    """

    def __init__(self, bins_file: str):
        """
        Initialize stratification.
        """
        super().__init__()

        self.bins: pd.Series = pd.read_csv(bins_file).iloc[:, 1]

    def get_types(self) -> List[str]:
        """
        Get all possible types.
        """
        return [f"bin{i}" for i in range(len(self.bins) - 1)]

    def get_type(self, variant: Union['cyvcf2.Variant', DummyVariant]) -> str:
        """
        Get type.
        """
        try:
            rho = variant.INFO['Recombination_intensity']
            return f"bin{sum(rho >= self.bins) - 1}"
        except KeyError:
            raise fd.io_handlers.NoTypeException()


@pytest.mark.inference
class ManuscriptTestCase(TestCase):
    """
    The code listings of the manuscript.
    """

    def test_base_inference_and_plots(self):
        """
        Listings 1 and 2: basic inference with bootstraps and the four summary plots.
        """
        inf = fd.BaseInference(
            sfs_neut=fd.Spectrum([66200, 410, 120, 60, 42, 43, 52, 65, 0]),
            sfs_sel=fd.Spectrum([281937, 600, 180, 87, 51, 43, 49, 61, 0]),
            model=fd.GammaExpParametrization(),
            n_runs=2,
            n_bootstraps=2,
            do_bootstrap=True
        )

        inf.run()

        axs = plt.subplots(2, 2, figsize=(11, 7))[1].flatten()

        types = ['neutral', 'selected']
        inf.plot_sfs_comparison(ax=axs[0], show=False, sfs_types=types)
        inf.plot_sfs_comparison(ax=axs[1], show=False, colors=['C1', 'C5'])
        inf.plot_inferred_parameters(ax=axs[2], show=False)
        inf.plot_discretized(ax=axs[3], show=False)

    def test_parser(self):
        """
        Listing 3: parse a degeneracy-stratified SFS with ancestral alleles from one outgroup.
        """
        p = fd.Parser(
            n=8,
            vcf=BETULA + "biallelic.with_outgroups.subset.10000.vcf.gz",
            fasta=BETULA + "genome.subset.1000.fasta.gz",
            gff=BETULA + "genome.gff.gz",
            target_site_counter=fd.TargetSiteCounter(
                n_target_sites=350000
            ),
            annotations=[
                fd.DegeneracyAnnotation(),
                fd.MaximumLikelihoodAncestralAnnotation(
                    outgroups=["ERR2103730"],
                    max_sites=300
                )
            ],
            stratifications=[fd.DegeneracyStratification()]
        )

        spectra: fd.Spectra = p.parse()

        spectra.plot(show=False)

        assert set(spectra.types) == {'neutral', 'selected'}

    def test_covariates_from_config(self):
        """
        Listings 4 and 5: covariate inference from the arabidopsis config, LRT and plots.
        """
        config = fd.Config.from_file("resources/configs/arabidopsis/covariates_example.yaml")
        config.update(n_runs=1, n_bootstraps=2)

        inf = fd.JointInference.from_config(config)

        inf.run()

        p = inf.perform_lrt_covariates()

        _, axs = plt.subplots(1, 2, figsize=(10.5, 3.5))
        p_str = f"p = {p:.1e}" if p >= 1e-100 else "p < 1e-100"

        inf.plot_covariate(ax=axs[0], xlabel='RSA', show=False)
        inf.plot_discretized(
            title=f"DFE comparison, " + p_str, ax=axs[1],
            show_marginals=False, show=False
        )


@pytest.mark.inference
class BookChapterVCFTestCase(TestCase):
    """
    The VCF scripts of the book chapter's workflow, run on the betula subset.
    """

    #: Temporary directory holding the outputs and the recombination files
    tmp: str

    #: Ingroup samples, in the chapter's ``samples.csv`` layout
    samples: pd.DataFrame

    @classmethod
    def setUpClass(cls):
        """
        Write the sample table, a genetic map and its recombination intensity bins.
        """
        cls.tmp = tempfile.mkdtemp()

        reader = cyvcf2.Reader(BETULA + "biallelic.with_outgroups.subset.10000.vcf.gz")
        ingroups = [s for s in reader.samples if s not in OUTGROUPS]
        reader.close()

        cls.samples = pd.DataFrame(dict(id=ingroups, species='betula_pendula'))

        rho = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]

        pd.DataFrame(dict(
            Chr=['Contig0', 'Contig1', 'Contig15'] * 2,
            Begin=[0] * 3 + [10 ** 5] * 3,
            End=[10 ** 5] * 3 + [10 ** 9] * 3,
            cMperMb=rho
        )).to_csv(f"{cls.tmp}/rho_map.csv", sep='\t', index=False)

        pd.DataFrame(np.quantile(rho, np.linspace(0, 1, 4))).to_csv(f"{cls.tmp}/rho_bins.csv")

    @classmethod
    def tearDownClass(cls):
        """
        Remove the temporary directory.
        """
        shutil.rmtree(cls.tmp)

    def test_annotate_and_filter(self):
        """
        ``annotate_ancestral_allele.py``, ``annotate_degeneracy.py``, ``annotate_recombination_intensity.py``,
        ``filter_vcf_snps.py`` and ``determine_ancestral_mismatches.py``, chained as in the workflow.
        """
        anc = fd.MaximumLikelihoodAncestralAnnotation(
            model=fd.K2SubstitutionModel(
                fix_transition_transversion_ratio=True
            ),
            outgroups=OUTGROUPS,
            ingroups=self.samples,
            max_sites=300,
            n_target_sites=3e6,
            prior=fd.AdaptivePolarizationPrior()
        )

        fd.Annotator(
            annotations=[anc],
            vcf=BETULA + "biallelic.with_outgroups.subset.10000.vcf.gz",
            fasta=BETULA + "genome.subset.1000.fasta.gz",
            output=f"{self.tmp}/anc.vcf.gz",
        ).annotate()

        anc.to_file(f"{self.tmp}/anc.json")
        anc.get_observed_transition_transversion_ratio()
        anc.get_outgroup_divergence()

        fd.Annotator(
            annotations=[fd.DegeneracyAnnotation()],
            vcf=f"{self.tmp}/anc.vcf.gz",
            gff=BETULA + "genome.gff.gz",
            fasta=BETULA + "genome.subset.1000.fasta.gz",
            output=f"{self.tmp}/anc.deg.vcf.gz"
        ).annotate()

        fd.Annotator(
            annotations=[RecombinationIntensityAnnotation(f"{self.tmp}/rho_map.csv", f"{self.tmp}/rho_bins.out.csv")],
            vcf=f"{self.tmp}/anc.deg.vcf.gz",
            gff=BETULA + "genome.gff.gz",
            fasta=BETULA + "genome.subset.1000.fasta.gz",
            output=f"{self.tmp}/anc.deg.rho.vcf.gz"
        ).annotate()

        fd.Filterer(
            vcf=f"{self.tmp}/anc.vcf.gz",
            filtrations=[
                fd.ExistingOutgroupFiltration(OUTGROUPS),
                fd.SNPFiltration(include_samples=self.samples)
            ],
            output=f"{self.tmp}/anc.filtered.vcf.gz"
        ).filter()

    def test_parse_plain(self):
        """
        ``parse_sfs_plain.py``: folded SFS including non-polarized sites.
        """
        sfs: fd.Spectra = fd.Parser(
            n=20,
            vcf=BETULA + "biallelic.with_outgroups.subset.10000.vcf.gz",
            skip_non_polarized=False,
            include_samples=self.samples.id
        ).parse()

        sfs.fold().plot(show=False)
        sfs.to_file(f"{self.tmp}/sfs.plain.csv")

    def test_parse_unfolded(self):
        """
        ``parse_sfs_unfolded.py`` and ``parse_sfs_unfolded_deg.py``.
        """
        for stratifications in [[], [fd.DegeneracyStratification()]]:
            sfs: fd.Spectra = fd.Parser(
                n=20,
                vcf=POLARIZED,
                include_samples=self.samples.id,
                polarize_probabilistically=True,
                stratifications=stratifications
            ).parse()

            sfs.plot(show=False)

    def test_parse_complete_and_by_contig(self):
        """
        ``parse_sfs_complete.py`` and ``parse_sfs_chr.py``: monomorphic sites, optionally stratified by contig.
        """
        for contigs in [None, ['Contig0', 'Contig1']]:
            p = fd.Parser(
                n=20,
                vcf=POLARIZED,
                fasta=BETULA + "genome.subset.1000.fasta.gz",
                gff=BETULA + "genome.gff.gz",
                include_samples=self.samples.id,
                stratifications=[fd.DegeneracyStratification()] + (
                    [fd.ContigStratification(contigs)] if contigs else []
                ),
                filtrations=([fd.ContigFiltration(contigs)] if contigs else []) + [
                    fd.SNPFiltration(),
                    fd.ExistingOutgroupFiltration(OUTGROUPS)
                ],
                target_site_counter=fd.TargetSiteCounter(
                    n_target_sites=2e7,
                    n_samples=1e4
                ),
                annotations=[fd.DegeneracyAnnotation()],
                polarize_probabilistically=True
            )

            sfs: fd.Spectra = p.parse()
            sfs.plot(show=False)

    def test_parse_random_and_recombination(self):
        """
        ``parse_sfs_random.py`` and ``parse_sfs_recombination.py``: random and custom recombination strata.
        """
        for stratification in [
            fd.RandomStratification(n_bins=10),
            RecombinationIntensityStratification(f"{self.tmp}/rho_bins.csv")
        ]:
            p = fd.Parser(
                n=20,
                vcf=POLARIZED,
                fasta=BETULA + "genome.subset.1000.fasta.gz",
                gff=BETULA + "genome.gff.gz",
                include_samples=self.samples.id,
                stratifications=[
                    fd.DegeneracyStratification(),
                    stratification
                ],
                filtrations=[
                    fd.SNPFiltration(),
                    fd.ExistingOutgroupFiltration(OUTGROUPS)
                ],
                target_site_counter=fd.TargetSiteCounter(
                    n_target_sites=2e7,
                    n_samples=1e4
                ),
                annotations=[
                    RecombinationIntensityAnnotation(f"{self.tmp}/rho_map.csv"),
                    fd.DegeneracyAnnotation()
                ],
                polarize_probabilistically=True
            )

            spectra: fd.Spectra = p.parse()
            spectra.plot(show=False)


@pytest.mark.inference
class BookChapterInferenceTestCase(TestCase):
    """
    The inference scripts of the book chapter's workflow, on the spectra it produced.
    """

    def test_infer_dfe(self):
        """
        ``infer_dfe.py``.
        """
        spectra = fd.Spectra.from_file(CHAPTER + 'sfs.csv')

        inf = fd.BaseInference(
            sfs_neut=spectra['neutral'],
            sfs_sel=spectra['selected'],
            fixed_params={'all': {'h': 0.5, 'eps': 0, 'p_b': 0, 'S_b': 1}},
            n_runs=2,
            n_bootstraps=2,
        )

        inf.run()

        inf.plot_discretized(show=False)
        inf.plot_sfs_comparison(show=False)

    def test_compare_nested(self):
        """
        ``compare_nested.py``.
        """
        spectra = fd.Spectra.from_file(CHAPTER + 'sfs.csv')

        inf = fd.BaseInference(
            sfs_neut=spectra['neutral'],
            sfs_sel=spectra['selected'],
            fixed_params={'all': {'h': 0.5}},
            n_runs=1,
            n_bootstraps=2
        )

        inf_sub = fd.BaseInference(
            sfs_neut=spectra['neutral'],
            sfs_sel=spectra['selected'],
            fixed_params={'all': {'h': 0.5, 'eps': 0}},
            n_runs=1,
            n_bootstraps=2
        )

        inf_sub.compare_nested(inf)

        fd.BaseInference(
            sfs_neut=spectra['neutral'],
            sfs_sel=spectra['selected'],
            fixed_params={'all': {'h': 0.5, 'p_b': 0, 'S_b': 1}},
            n_runs=1,
            n_bootstraps=2
        ).compare_nested(inf)

        inf.plot_nested_models(show=False)

    def test_compare_parametrizations(self):
        """
        ``compare_parametrization.py``.
        """
        spectra = fd.Spectra.from_file(CHAPTER + 'sfs.csv')

        parametrizations = [
            fd.GammaExpParametrization(),
            fd.DiscreteFractionalParametrization(),
            fd.GammaDiscreteParametrization(),
            fd.DisplacedGammaParametrization()
        ]

        inferences = [fd.BaseInference(
            sfs_neut=spectra['neutral'],
            sfs_sel=spectra['selected'],
            fixed_params={'all': {'h': 0.5, 'eps': 0}},
            model=p,
            n_runs=1,
            do_bootstrap=False
        ) for p in parametrizations]

        [inf.run() for inf in inferences]

        fd.Inference.plot_discretized(
            inferences=inferences,
            labels=[type(p).__name__ for p in parametrizations],
            intervals=[-np.inf, -100, -10, -1, 1, np.inf],
            show=False
        )

    def test_infer_joint_dfe(self):
        """
        ``infer_joint_dfe.py``.
        """
        spectra = fd.Spectra.from_file(CHAPTER + 'sfs.chr.csv')

        for fixed, shared in [
            ({'h': 0.5, 'eps': 0, 'p_b': 0, 'S_b': 1}, ['S_d']),
            ({'h': 0.5, 'eps': 0}, ['S_d']),
            ({'h': 0.5, 'eps': 0, 'p_b': 0, 'S_b': 1}, ['b'])
        ]:
            inf = fd.JointInference(
                sfs_neut=spectra['neutral.*'].rename(['chr1', 'chr2']),
                sfs_sel=spectra['selected.*'].rename(['chr1', 'chr2']),
                fixed_params={'all': fixed},
                shared_params=[fd.SharedParams(types=['chr1', 'chr2'], params=shared)],
                n_runs=1,
                n_bootstraps=2
            )

            inf.run()
            inf.plot_discretized(show=False)
            inf.perform_lrt_shared()

    def test_infer_covariate_dfe(self):
        """
        ``infer_covariate_dfe.py``, whose code ``infer_random_dfe.py`` repeats on ``sfs.random.csv``.
        """
        spectra = fd.Spectra.from_file(CHAPTER + 'sfs.rho.csv')
        bins = pd.read_csv(CHAPTER + 'rho_bins.csv').iloc[:, 1].to_numpy()
        values = np.log10((bins[:-1] + bins[1:]) / 2)
        types = spectra.merge_groups(level=1).types
        covariate = fd.Covariate('S_d', dict(zip(types, values)))

        inf = fd.JointInference(
            sfs_neut=spectra['neutral.*'].merge_groups(level=1),
            sfs_sel=spectra['selected.*'].merge_groups(level=1),
            fixed_params={'all': {'h': 0.5, 'eps': 0, 'p_b': 0, 'S_b': 1}},
            shared_params=[fd.SharedParams(['b', 'S_d'])],
            covariates=[covariate],
            n_bootstrap_retries=10,
            n_runs=1,
            n_bootstraps=2
        )

        inf.run()
        inf.plot_discretized(show=False)

        inf.bootstraps.select_dtypes('number').mean()['bin0.c0']

        for inferences in [inf.marginal_inferences, inf.joint_inferences]:
            {k: v.bootstraps.select_dtypes('number').mean()['S_d'] for k, v in inferences.items() if k != 'all'}

        inf.perform_lrt_covariates()
        inf.plot_covariate(show=False)

    def test_plot_sfs(self):
        """
        ``plot_sfs.py``.
        """
        fd.Spectra.from_file(CHAPTER + 'sfs.rho.csv').plot(show=False)
