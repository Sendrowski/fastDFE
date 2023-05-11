"""
VCF annotators.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2023-05-09"

import logging
from abc import abstractmethod, ABC
from typing import Iterable, List

import numpy as np
from cyvcf2 import Variant, Writer, VCF
from collections import Counter

from fastdfe.vcf import VCFHandler, get_called_bases

# get logger
logger = logging.getLogger('fastdfe')


class Annotation:

    def __init__(self):
        """
        Create a new annotation instance.
        """
        #: The annotator.
        self.annotator: Annotator | None = None

    def provide_context(self, annotator: 'Annotator'):
        """
        Provide context by passing the annotator. This should be called before the annotation starts.

        :param annotator: The annotator.
        """
        self.annotator = annotator

    def add_info(self, reader: VCF):
        """
        Add info fields to the header.
        """
        pass

    def annotate_site(self, variant: Variant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        pass


class AncestralAlleleAnnotation(Annotation, ABC):
    """
    Base class for ancestral allele annotation.
    """

    def add_info(self, reader: VCF):
        """
        Add info fields to the header.
        """
        reader.add_info_to_header({
            'ID': self.annotator.info_ancestral,
            'Number': '.',
            'Type': 'Character',
            'Description': 'Ancestral Allele'
        })


class MaximumParsimonyAnnotation(AncestralAlleleAnnotation):
    """
    Annotation of ancestral alleles using maximum parsimony.
    """

    def annotate_site(self, variant: Variant):
        """
        Annotate a single site.

        :param variant: The variant to annotate.
        :return: The annotated variant.
        """
        # get the called bases
        bases = get_called_bases(variant)

        # get the major allele
        major_allele = Counter(bases).most_common(1)[0][0]

        # set the ancestral allele
        variant.INFO[self.annotator.info_ancestral] = major_allele


class Annotator(VCFHandler):
    """
    Annotator base class.
    """

    def __init__(
            self,
            vcf: str,
            output: str,
            annotations: List[Annotation],
            info_ancestral: str = 'AA',
            max_sites: int = np.inf,
            seed: int | None = 0
    ):
        """
        Create a new annotator instance.

        :param vcf: The path to the VCF file
        :param output: The path to the output file
        :param annotations: The annotations to apply.
        :param info_ancestral: The tag in the INFO field that contains the ancestral allele
        :param max_sites: Maximum number of sites to consider
        :param seed: Seed for the random number generator
        """
        super().__init__(
            vcf=vcf,
            info_ancestral=info_ancestral,
            max_sites=max_sites,
            seed=seed
        )

        self.output: str = output

        self.annotations: List[Annotation] = annotations

    def annotate(self):
        """
        Annotate the VCF file.
        """
        # count the number of sites
        self.n_sites = self.count_lines_vcf()

        # create the reader
        reader = VCF(self.vcf)

        # provide annotator to annotations and add info fields
        for annotation in self.annotations:
            annotation.provide_context(self)
            annotation.add_info(reader)

        # create the writer
        writer = Writer(self.output, reader)

        # iterate over the sites
        for i, variant in enumerate(self.get_sites(reader)):

            # stop if max_sites was reached
            if i >= self.max_sites:
                break

            # apply annotations
            for annotation in self.annotations:
                annotation.annotate_site(variant)

            # write the variant
            writer.write_record(variant)
