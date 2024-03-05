from testing import prioritize_installed_packages

# logging.getLogger('fastdfe').setLevel(logging.DEBUG)

prioritize_installed_packages()

from testing import TestCase


class FastDFETestCase(TestCase):
    """
    Test the annotators.
    """

    def test_load_fastdfe(self):
        """
        Test loading fastdfe.
        """
        # noinspection PyUnresolvedReferences
        import fastdfe as fd
