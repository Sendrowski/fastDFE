import logging

from cyvcf2 import VCF, Writer

from testing import prioritize_installed_packages

prioritize_installed_packages()

from unittest import TestCase

from fastdfe import Annotator, MaximumParsimonyAnnotation, Parser

logging.getLogger('fastdfe').setLevel(logging.DEBUG)


class AnnotatorTestCase(TestCase):
    """
    Test the annotators.
    """

    vcf_file = 'resources/genome/betula/biallelic.subset.10000.vcf.gz'

    def test_maximum_parsimony_annotator_different_info_field(self):
        """
        Test the maximum parsimony annotator.
        """
        ann = Annotator(
            vcf=self.vcf_file,
            output='scratch/test_maximum_parsimony_annotator.vcf',
            annotations=[MaximumParsimonyAnnotation()],
            info_ancestral='BB'
        )

        ann.annotate()

        Parser(self.vcf_file, 20).parse().plot(title="Original")
        Parser(ann.output, 20).parse().plot(title="Annotated")

