"""
A VCF parser that can be used to extract the site frequency spectrum (SFS) from a VCF file.
Stratifying the SFS is supported by providing a list of :class:`Stratification` instances.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-03-26"

import itertools
import logging
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Callable, Literal, Optional, Iterable, Dict, cast

import numpy as np
from Bio.SeqRecord import SeqRecord
from cyvcf2 import Variant, VCF

from .annotation import Annotation, Annotator, SynonymyAnnotation
from .bio_handlers import bases, get_called_bases, count_no_type, FASTAHandler, VCFHandler, NoTypeException
from .filtration import Filtration, PolyAllelicFiltration
from .spectrum import Spectra

# logger
logger = logging.getLogger('fastdfe')


class Stratification(ABC):
    """
    Abstract class for Stratifying the SFS by determining a site's type based on its properties.
    """

    #: Parser instance
    parser: Optional['Parser'] = None

    #: The number of sites that didn't have a type.
    n_no_type: int = 0

    def __init__(self):
        """
        Create instance.
        """
        self.logger = logger.getChild(self.__class__.__name__)

    def _setup(self, parser: 'Parser'):
        """
        Provide the stratification with some context by specifying the parser.
        This should be done before calling :meth:`get_type`.

        :param parser: The parser
        """
        self.parser = parser

    def _teardown(self):
        """
        Perform any necessary post-processing. This method is called after the actual stratification.
        """
        n_total = self.parser.n_sites - self.parser.n_skipped + self.n_no_type
        n_valid = n_total - self.n_no_type

        self.logger.info(f"Number of sites with valid type: {n_valid} / {n_total}")

    @abstractmethod
    def get_type(self, variant: Variant) -> Optional[str]:
        """
        Get type of given Variant. Only the types
        given by :meth:`get_types()` are valid, or ``None`` if
        no type could be determined.

        :param variant: The vcf site
        :return: Type of the variant
        """
        pass

    @abstractmethod
    def get_types(self) -> List[str]:
        """
        Get all possible types.

        :return: List of types
        """
        pass


class BaseContextStratification(Stratification, FASTAHandler):
    """
    Stratify the SFS by the base context of the mutation. The number of flanking bases
    can be configured. Note that we attempt to take the ancestral allele as the
    middle base. If ``skip_non_polarized`` is set to ``True``, we skip sites
    that are not polarized, otherwise we use the reference allele as the middle base.
    """

    def __init__(
            self,
            fasta_file: str,
            n_flanking: int = 1,
            aliases: Dict[str, List[str]] = {},
            cache: bool = True
    ):
        """
        Create instance. Note that we require a fasta file to be specified
        for base context to be able to be inferred

        :param fasta_file: The fasta file path, possibly gzipped or a URL
        :param n_flanking: The number of flanking bases
        :param aliases: Dictionary of aliases for the contigs in the VCF file, e.g. ``{'chr1': ['1']}``.
            This is used to match the contig names in the VCF file with the contig names in the FASTA file and GFF file.
        :param cache: Whether to cache files that are downloaded from URLs
        """
        Stratification.__init__(self)

        FASTAHandler.__init__(self, fasta_file, cache=cache)

        #: The number of flanking bases
        self.n_flanking: int = n_flanking

        #: Aliases for the contigs in the VCF file
        self.aliases: Dict[str, List[str]] = aliases

        #: The current contig
        self.contig: Optional[SeqRecord] = None

    @count_no_type
    def get_type(self, variant: Variant) -> str:
        """
        Get the base context for a given mutation

        :param variant: The vcf site
        :return: Base context of the mutation
        """
        pos = variant.POS - 1

        # get the ancestral allele
        aa = self.parser._get_ancestral(variant)

        # get aliases
        aliases = self.get_aliases(variant.CHROM, self.aliases)

        # check if contig is up-to-date
        if self.contig is None or self.contig.id not in aliases:
            self.logger.debug(f"Fetching contig '{variant.CHROM}'.")

            # fetch contig
            self.contig = self.get_contig(aliases)

        # check if position is valid
        if pos < 0 or pos >= len(self.contig):
            raise NoTypeException("Invalid position: Position must be within the bounds of the sequence.")

        # get upstream bases
        upstream_start = max(0, pos - self.n_flanking)
        upstream_bases = str(self.contig.seq[upstream_start:pos])

        # get downstream bases
        downstream_end = min(len(self.contig), pos + self.n_flanking + 1)
        downstream_bases = str(self.contig.seq[pos + 1:downstream_end])

        return f"{upstream_bases}{aa}{downstream_bases}"

    def get_types(self) -> List[str]:
        """
        Create all possible base contexts.

        :return: List of contexts
        """
        return [''.join(c) for c in itertools.product(bases, repeat=2 * self.n_flanking + 1)]


class BaseTransitionStratification(Stratification):
    """
    Stratify the SFS by the base transition of the mutation, i.e., ``A>T``.
    Should be used with ``n_target_sites`` since we can't infer the type of monomorphic sites.
    """

    @count_no_type
    def get_type(self, variant: Variant) -> str:
        """
        Get the base transition for the given variant.

        :param variant: The vcf site
        :return: Base transition
        :raises NoTypeException: if not type could be determined
        """
        if variant.is_snp:
            ancestral = self.parser._get_ancestral(variant)

            derived = variant.REF if variant.REF != ancestral else variant.ALT[0]

            if ancestral in bases and derived in bases and ancestral != derived:
                return f"{ancestral}>{derived}"

            raise NoTypeException("Not a valid base transition.")

        raise NoTypeException("Site is not a SNP.")

    def get_types(self) -> List[str]:
        """
        Get all possible base transitions.

        :return: List of contexts
        """
        return ['>'.join([a, b]) for a in bases for b in bases if a != b]


class AncestralBaseStratification(Stratification):
    """
    Stratify the SFS by the base context of the mutation: the reference base.
    If ``skip_non_polarized`` is set to ``True``, we skip sites
    that are not polarized, otherwise we use the reference allele as ancestral base.
    By default, we use the ``AA`` tag to determine the ancestral allele.

    Any subclass of :class:`~fastdfe.parser.AncestralAnnotation` can be used to annotate the ancestral allele.
    """

    @count_no_type
    def get_type(self, variant: Variant) -> str:
        """
        Get the type which is the reference allele.

        :param variant: The vcf site
        :return: reference allele
        """
        return self.parser._get_ancestral(variant)

    def get_types(self) -> List[str]:
        """
        The possible base types.

        :return: List of contexts
        """
        return bases


class TransitionTransversionStratification(BaseTransitionStratification):
    """
    Stratify the SFS by whether we have a transition or transversion.
    Should be used with ``n_target_sites`` since we can't infer the type of monomorphic sites.
    """

    @count_no_type
    def get_type(self, variant: Variant) -> str:
        """
        Get the mutation type (transition or transversion) for a given mutation.

        :param variant: The vcf site
        :return: Mutation type
        """
        if variant.is_snp:

            if variant.ALT[0] not in bases:
                raise NoTypeException("Invalid alternate allele: Alternate allele must be a valid base.")

            if (variant.REF, variant.ALT[0]) in [('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')]:
                return "transition"
            else:
                return "transversion"

        raise NoTypeException("Site is not a SNP.")

    def get_types(self) -> List[str]:
        """
        All possible mutation types (transition and transversion).

        :return: List of mutation types
        """
        return ["transition", "transversion"]


class DegeneracyStratification(Stratification):
    """
    Stratify SFS by degeneracy. We only consider sides which 4-fold degenerate (neutral) or
    0-fold degenerate (selected) which facilitates counting.

    :class:`~fastdfe.annotation.DegeneracyAnnotation` can be used to annotate the degeneracy of a site.
    """

    def __init__(
            self,
            custom_callback: Callable[[Variant], str] = None,
    ):
        """
        Initialize the stratification.

        :param custom_callback: Custom callback to determine the type of mutation
        """
        super().__init__()

        #: Custom callback to determine the degeneracy of mutation
        self.get_degeneracy = custom_callback if custom_callback is not None else self._get_degeneracy_default

    @staticmethod
    def _get_degeneracy_default(variant: Variant) -> Optional[Literal['neutral', 'selected']]:
        """
        Get degeneracy based on 'Degeneracy' tag.

        :param variant: The vcf site
        :return: Type of the mutation
        """
        degeneracy = variant.INFO.get('Degeneracy')

        if degeneracy is None:
            raise NoTypeException("No degeneracy tag found.")
        else:
            if degeneracy == 4:
                return 'neutral'

            if degeneracy == 0:
                return 'selected'

            raise NoTypeException(f"Degeneracy tag has invalid value: '{degeneracy}' at {variant.CHROM}:{variant.POS}")

    @count_no_type
    def get_type(self, variant: Variant) -> Literal['neutral', 'selected']:
        """
        Get the degeneracy.

        :param variant: The vcf site
        :return: Type of the mutation
        :raises NoTypeException: If the mutation is not synonymous or non-synonymous
        """
        return self.get_degeneracy(variant)

    def get_types(self) -> List[str]:
        """
        Get all possible degeneracy type (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']


class SynonymyStratification(Stratification):
    """
    Stratify SFS by synonymy (neutral or selected).

    :class:`~fastdfe.annotation.SynonymyAnnotation` can be used to annotate the synonymy of a site.
    """

    def get_types(self) -> List[str]:
        """
        Get all possible synonymy types (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']

    @count_no_type
    def get_type(self, variant: Variant) -> Literal['neutral', 'selected']:
        """
        Get the synonymy using the custom synonymy annotation.

        :param variant: The vcf site
        :return: Type of the mutation, either ``neutral`` or ``selected``
        """
        synonymy = variant.INFO.get('Synonymy')

        if synonymy is None:
            raise NoTypeException("No synonymy tag found.")
        else:
            if synonymy == 1:
                return 'neutral'

            if synonymy == 0:
                return 'selected'

            raise NoTypeException(f"Synonymy tag has invalid value: '{synonymy}' at {variant.CHROM}:{variant.POS}")


class VEPStratification(SynonymyStratification):
    """
    Stratify SFS by synonymy (neutral or selected) based on annotation provided by VEP.
    Note that since we cannot determine the synonymy for monomorphic sites, this should be used together
    with ``n_target_sites``.
    """

    #: The tag used by VEP to annotate the synonymy
    info_tag = 'CSQ'

    def get_types(self) -> List[str]:
        """
        Get all possible synonymy types (``neutral`` and ``selected``).

        :return: List of contexts
        """
        return ['neutral', 'selected']

    @count_no_type
    def get_type(self, variant: Variant) -> Literal['neutral', 'selected']:
        """
        Get the synonymy of a site.

        :param variant: The vcf site
        :return: Type of the mutation, either ``neutral`` or ``selected``
        """
        synonymy = variant.INFO.get(self.info_tag, '')

        if 'synonymous_variant' in synonymy:
            return 'neutral'

        if 'missense_variant' in synonymy:
            return 'selected'

        raise NoTypeException(f"Synonymy tag has invalid value: '{synonymy}' at {variant.CHROM}:{variant.POS}")


class SnpEffStratification(VEPStratification):
    """
    Stratify SFS by synonymy (neutral or selected) based on annotation provided by SnpEff.
    Note that since we cannot determine the synonymy for monomorphic sites, this should be used together
    with ``n_target_sites``.
    """

    #: The tag used by SnpEff to annotate the synonymy
    info_tag = 'ANN'


class GenomePositionDependentStratification(Stratification, ABC):
    pass


class ContigStratification(GenomePositionDependentStratification):
    """
    Stratify SFS by contig.
    """

    def get_type(self, variant: Variant) -> str:
        """
        Get the contig.

        :param variant: The vcf site
        :return: The contig name
        """
        return variant.CHROM

    def get_types(self) -> List[str]:
        """
        Get all possible contig type.

        :return: List of contexts
        """
        return list(self.parser.reader.seqnames)


class ChunkedStratification(GenomePositionDependentStratification):
    """
    Stratify SFS by creating ``n`` chunks of roughly equal size.
    """

    def __init__(self, n_chunks: int):
        """
        Initialize the stratification.

        :param n_chunks: Number of sites per window
        """
        super().__init__()

        #: Number of chunks
        self.n_chunks: int = int(n_chunks)

        #: List of chunk sizes
        self.chunk_sizes: Optional[List[int]] = None

        #: Number of sites seen so far
        self.counter: int = 0

    def _setup(self, parser: 'Parser'):
        """
        Set up the stratification.

        :param parser: The parser
        """
        super()._setup(parser)

        # compute base chunk size and remainder
        base_chunk_size, remainder = divmod(parser.n_sites, self.n_chunks)

        # create list of chunk sizes
        self.chunk_sizes = [base_chunk_size + (i < remainder) for i in range(self.n_chunks)]

    def get_types(self) -> List[str]:
        """
        Get all possible window types.

        :return: List of contexts
        """
        return [f'chunk{i}' for i in range(self.n_chunks)]

    def get_type(self, variant: Variant) -> str:
        """
        Get the type.

        :param variant: The vcf site
        :return: The type
        """
        # find the index of the chunk to which the current site belongs
        chunk_index = next(i for i, size in enumerate(self.chunk_sizes) if self.counter < sum(self.chunk_sizes[:i + 1]))

        # get the type
        t = f'chunk{chunk_index}'

        # update the counter
        self.counter += 1

        return t


class Parser(VCFHandler):
    """
    Parse site-frequency spectra from VCF files.

    By default, the parser looks at the ``AA`` tag in the VCF file's info field to retrieve
    the correct polarization. Sites for which this tag is not well-defined are by default
    included. Note that non-polarized frequency spectra provide little information on the
    distribution of beneficial mutations.

    We can also annotate the SFS with additional information, such as the degeneracy of the
    sites and their ancestral alleles. This is done by providing a list of annotations to
    the parser. The annotations are applied in the order they are provided.

    The parser also allows to filter sites based on their annotations. This is done by
    providing a list of filtrations to the parser. By default, we filter out poly-allelic
    sites which is highly recommended as some stratifications assume sites to be at most bi-allelic.

    Example usage:

    ::

        import fastdfe as fd

        # parse selected and neutral SFS from human chromosome 21
        p = fd.Parser(
            vcf="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
                "1000_genomes_project/release/20181203_biallelic_SNV/"
                "ALL.chr21.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
            n=10,
            annotations=[
                fd.DegeneracyAnnotation(
                    fasta_file="http://ftp.ensembl.org/pub/release-109/fasta/homo_sapiens/"
                               "dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz",
                    gff_file="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                             "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
                    aliases=dict(chr21=['21'])
                ),
                fd.MaximumParsimonyAncestralAnnotation()
            ],
            filtrations=[
                fd.CodingSequenceFiltration(
                    gff_file="http://ftp.ensembl.org/pub/release-109/gff3/homo_sapiens/"
                             "Homo_sapiens.GRCh38.109.chromosome.21.gff3.gz",
                    aliases=dict(chr21=['21'])
                )
            ],
            stratifications=[fd.DegeneracyStratification()],
        )

        sfs = p.parse()

        sfs.plot()

    """

    def __init__(
            self,
            vcf: str | Iterable[Variant],
            n: int,
            info_ancestral: str = 'AA',
            skip_non_polarized: bool = False,
            stratifications: List[Stratification] = [],
            annotations: List[Annotation] = [],
            filtrations: List[Filtration] = [PolyAllelicFiltration()],
            samples: List[str] = None,
            max_sites: int = np.inf,
            n_target_sites: int = None,
            seed: int | None = 0,
            cache: bool = True
    ):
        """
        Initialize the parser.

        :param vcf: The path to the VCF file or an iterable of variants, can be gzipped, urls are also supported
        :param n: The number of individuals in the sample. We down-sample to this number by drawing without replacement.
            Sites with fewer than ``n`` individuals are skipped.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param skip_non_polarized: Whether to skip sites that are not polarized, i.e., without a valid info tag
            providing the ancestral allele
        :param stratifications: List of stratifications to use
        :param annotations: List of annotations to use
        :param filtrations: List of filtrations to use.
        :param samples: List of samples to use. If ``None``, all samples are used.
        :param max_sites: Maximum number of sites to parse
        :param n_target_sites: The number of mutational target sites.
            Allows to adjust the number of monomorphic site count. Ideally, we obtain the SFS by
            parsing VCF files that contain both mono- and polymorphic sites. This is because for DFE inference, we
            require the number of monomorphic sites to calibrate the mutation rate. However, often, only polymorphic
            sites are available. In this case, we can use ``n_target_sites`` to extrapolate the number of monomorphic
            sites by looking at the relative number of polymorphic sites for each type. Note that the *total* number of
            mono- and polymorphic sites should be specified here. This often corresponds to the number of sites in
            coding regions over the sequence considered. If ``None``, and
            :class:`~fastdfe.annotation.SynonymyAnnotation` is used, the number
            of monomorphic sites is inferred dynamically from the length of the coding sequences considered.
            Alternatively, you can use :meth:`~fastdfe.annotation.Annotation.count_target_sites` to count the number of
            coding sites manually.
        :param seed: Seed for the random number generator. Use ``None`` for no seed.
        :param cache: Whether to cache files downloaded from urls
        """
        super().__init__(
            vcf=vcf,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed,
            cache=cache
        )

        #: The number of individuals in the sample
        self.n: int = int(n)

        #: The number of mutational target sites
        self.n_target_sites: int | None = n_target_sites

        #: The list of samples to use
        self.samples: List[str] | None = samples

        #: The mask of samples to use
        self.samples_mask: np.ndarray | None = None

        #: Whether to skip sites that are not polarized, i.e., without a valid info tag providing the ancestral allele
        self.skip_non_polarized: bool = skip_non_polarized

        #: List of stratifications to use
        self.stratifications: List[Stratification] = stratifications

        #: List of annotations to use
        self.annotations: List[Annotation] = annotations

        #: List of filtrations to use
        self.filtrations: List[Filtration] = filtrations

        #: The number of sites that were skipped for various reasons
        self.n_skipped: int = 0

        #: Dictionary of SFS indexed by joint type
        self.sfs: Dict[str, np.ndarray] = {}

        #: The VCF reader
        self.reader: Optional[VCF] = None

    def _get_ancestral(self, variant: Variant) -> str:
        """
        Determine the ancestral allele.

        :param variant: The vcf site
        :return: Ancestral allele
        :raises NoTypeException: If the site is not polarized and ``skip_non_polarized`` is ``True``
        """
        if variant.is_snp:
            # obtain ancestral allele
            aa = variant.INFO.get(self.info_ancestral)

            # return the ancestral allele if it is a valid base
            if aa in bases:
                return aa

            # if we skip non-polarized sites, we raise an error here
            if self.skip_non_polarized:
                raise NoTypeException("No valid AA tag found so we skip the site")

        # if we don't skip non-polarized sites, or if the site is not an SNP
        # we return the reference allele if valid
        if variant.REF in bases:
            return variant.REF

        # if the reference allele is not a valid base, we raise an error
        raise NoTypeException("Reference allele is not a valid base")

    def _create_sfs_dictionary(self) -> Dict[str, np.ndarray]:
        """
        Create an SFS dictionary initialized with all possible base contexts.

        :return: SFS dictionary
        """
        types = [s.get_types() for s in self.stratifications]

        # define the DNA bases
        contexts = ['.'.join(t) for t in itertools.product(*types)]

        # create dict
        sfs = {}
        for context in contexts:
            sfs[context] = np.zeros(self.n + 1)

        return sfs

    def _parse_site(self, variant: Variant):
        """
        Parse a single site.

        :param variant: The variant
        """
        if variant.is_snp:

            # obtain called bases
            genotypes = get_called_bases(variant.gt_bases[self.samples_mask])

            # number of samples
            n_samples = len(genotypes)

            # skip if not enough samples
            if n_samples < self.n:
                self.logger.debug(f'Skipping site due to too few samples at {variant.CHROM}:{variant.POS}.')
                self.n_skipped += 1
                return

            try:
                # determine ancestral allele
                aa = self._get_ancestral(variant)
            except NoTypeException:
                self.n_skipped += 1
                return

            # count called bases
            counter = Counter(genotypes)

            # determine ancestral allele count
            n_aa = counter[aa]

            # determine down-projected allele count
            k = self.rng.hypergeometric(ngood=n_samples - n_aa, nbad=n_aa, nsample=self.n)

        else:
            # if we don't have an SNP, we assume the reference allele to be the ancestral allele,
            # so the derived allele count is 0
            k = 0

        # try to obtain type
        try:
            # create joint type
            t = '.'.join([s.get_type(variant) for s in self.stratifications]) or 'all'

            # if the type is not in the dictionary, we add it
            if t not in self.sfs:
                self.sfs[t] = np.zeros(self.n + 1)

            # add count by 1
            self.sfs[t][k] += 1

        except NoTypeException as e:
            self.n_skipped += 1
            self.logger.debug(e)

    def _process_site(self, variant: Variant):
        """
        Handle a single site.

        :param variant: The variant
        """
        # filter the variant
        for filtration in self.filtrations:
            if not filtration.filter_site(variant):
                self.n_skipped += 1
                return

        # apply annotations
        for annotation in self.annotations:
            annotation.annotate_site(variant)

        # parse site
        self._parse_site(variant)

    def _teardown(self):
        """
        Teardown the parser.
        """
        for f in self.filtrations:
            f._teardown()

        for s in self.stratifications:
            s._teardown()

        for a in self.annotations:
            a._teardown()

    def _infer_monomorphic_counts(self):
        """
        Infer the number of monomorphic sites from the number of polymorphic sites.
        """
        # total number of polymorphic sites across all types
        n_polymorphic = np.sum([np.sum(c[1:-1]) for c in self.sfs.values()])

        for t, counts in self.sfs.items():
            self.sfs[t][0] = np.sum(counts[1:-1]) / n_polymorphic * (self.n_target_sites - n_polymorphic)
            self.sfs[t][-1] = 0

    def _update_target_sites(self):
        """
        Update the number of target sites based on the SynonymyAnnotation if present.
        """

        if self.n_target_sites is None:

            # find index of SynonymyAnnotation in list
            index = next((i for i, a in enumerate(self.annotations) if isinstance(a, SynonymyAnnotation)), -1)

            if index != -1:
                self.logger.info(f'Updating target sites based on SynonymyAnnotation.')

                self.n_target_sites = cast(SynonymyAnnotation, self.annotations[index]).n_target_sites

    def parse(self) -> Spectra:
        """
        Parse the VCF file.

        :return: The spectra for the different stratifications
        """
        # create reader
        self.reader = VCF(self.download_if_url(self.vcf))

        # count the number of sites
        self.n_sites = self.count_sites()

        # make parser available to stratifications
        for s in self.stratifications:
            s._setup(self)

        # create a string representation of the stratifications
        representation = '.'.join(['[' + ', '.join(s.get_types()) + ']' for s in self.stratifications]) or "[all]"

        # log the stratifications
        self.logger.info(f'Using stratification: {representation}.')

        # instantiate annotator to provide context to annotations
        ann = Annotator(
            vcf=self.vcf,
            max_sites=self.max_sites,
            seed=self.seed,
            info_ancestral=self.info_ancestral,
            annotations=[],
            output=''
        )

        # create samples mask
        if self.samples is None:
            self.samples_mask = np.ones(len(self.reader.samples)).astype(bool)
        else:
            self.samples_mask = np.isin(self.reader.samples, self.samples)

        # provide annotator to annotations and add info fields
        for annotation in self.annotations:
            annotation._setup(ann, self.reader)

        # touch all filtrations
        for f in self.filtrations:
            f._setup(self.reader)

        self.logger.info(f'Starting to parse.')

        # create progress bar
        with self.get_pbar() as pbar:

            for i, variant in enumerate(self.reader):

                # handle site
                self._process_site(variant)

                pbar.update()

                # explicitly stopping after ``n``sites fixes a bug with cyvcf2:
                # 'error parsing variant with `htslib::bcf_read` error-code: 0 and ret: -2'
                if i + 1 == self.n_sites or i + 1 == self.max_sites:
                    break

        # tear down all objects
        self._teardown()

        # close reader
        self.reader.close()

        # update target sites
        self._update_target_sites()

        # correct monomorphic counts if number of target sites is defined
        if self.n_target_sites is not None:
            self._infer_monomorphic_counts()

        if len(self.sfs) == 0:
            self.logger.warning(f"No sites were included in the spectra. If this is not expected, "
                                "please check that all components work as expected. You can do this by "
                                "setting the log level to DEBUG.")
        else:
            self.logger.info(f'Included {self.n_sites - self.n_skipped} out of {self.n_sites} sites in total.')

        return Spectra(self.sfs).sort_types()
