import os
import tempfile
import uuid
from unittest.mock import patch, MagicMock

from fastdfe.io_handlers import FileHandler, download_if_url
from testing import TestCase


def _fake_response(content: bytes) -> MagicMock:
    """
    Build a stand-in for the streaming ``requests`` response used by ``download_file`` so the
    download path can be exercised without any network access.
    """
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.headers = {'content-length': str(len(content))}
    resp.iter_content.return_value = iter([content])
    return resp


class IOHandlerURLTestCase(TestCase):
    """
    Fast tests for the URL-handling layer in :mod:`fastdfe.io_handlers` (URL detection,
    filename/hash derivation, downloading and caching). Network access is mocked, so these run in
    milliseconds and never touch a remote server -- unlike the heavy ``slow``-tier tests that pull
    whole chromosomes from Ensembl/Sanger/GitHub.
    """

    def setUp(self):
        # a unique URL per test keeps the cache path (a hash of the URL) fresh across runs
        self.url = f"https://example.com/data/{uuid.uuid4().hex}/file.vcf.gz"
        self.payload = b"##fileformat=VCFv4.2\nfoo\nbar\n"
        self.cached_path = (tempfile.gettempdir() + '/' +
                            FileHandler.hash(self.url) + '.' + FileHandler.get_filename(self.url))

    def tearDown(self):
        if os.path.exists(self.cached_path):
            os.remove(self.cached_path)

    def test_is_url_recognizes_remote_schemes(self):
        """URLs with a scheme and host are recognised as remote."""
        self.assertTrue(FileHandler.is_url("https://example.com/x.vcf.gz"))
        self.assertTrue(FileHandler.is_url("http://example.com/x.fa"))
        self.assertTrue(FileHandler.is_url("ftp://ftp.ensembl.org/pub/x.gff3.gz"))

    def test_is_url_rejects_local_paths(self):
        """Local paths (relative, absolute, bare filename) are not treated as URLs."""
        self.assertFalse(FileHandler.is_url("resources/genome/betula/genome.gff.gz"))
        self.assertFalse(FileHandler.is_url("/abs/path/file.vcf"))
        self.assertFalse(FileHandler.is_url("file.vcf"))

    def test_get_filename_from_url(self):
        """The cached filename is the basename of the URL path, ignoring any query string."""
        self.assertEqual("chr21.fa.gz",
                         FileHandler.get_filename("http://ftp.ensembl.org/pub/chr21.fa.gz"))
        self.assertEqual("data.vcf",
                         FileHandler.get_filename("https://host/dir/data.vcf?token=abc"))

    def test_hash_is_deterministic_and_distinct(self):
        """The URL hash is stable, collision-free for distinct inputs, and truncated to 12 chars."""
        self.assertEqual(FileHandler.hash("a"), FileHandler.hash("a"))
        self.assertNotEqual(FileHandler.hash("a"), FileHandler.hash("b"))
        self.assertEqual(12, len(FileHandler.hash("anything")))

    def test_download_file_writes_payload(self):
        """A URL download lands at the hashed cache path with the streamed content intact."""
        with patch('fastdfe.io_handlers.requests.get',
                   return_value=_fake_response(self.payload)) as mock_get:
            path = FileHandler.download_file(self.url)

        self.assertEqual(self.cached_path, path)
        self.assertTrue(os.path.exists(path))
        with open(path, 'rb') as f:
            self.assertEqual(self.payload, f.read())
        mock_get.assert_called_once()

    def test_download_file_uses_cache_on_second_call(self):
        """With caching on, a repeated download is served from disk without hitting the network."""
        with patch('fastdfe.io_handlers.requests.get',
                   return_value=_fake_response(self.payload)) as mock_get:
            FileHandler.download_file(self.url, cache=True)
            FileHandler.download_file(self.url, cache=True)

            mock_get.assert_called_once()

    def test_download_file_no_cache_redownloads(self):
        """With caching off, every call re-fetches even when the file is already on disk."""
        with patch('fastdfe.io_handlers.requests.get',
                   side_effect=lambda *a, **k: _fake_response(self.payload)) as mock_get:
            FileHandler.download_file(self.url, cache=False)
            FileHandler.download_file(self.url, cache=False)

            self.assertEqual(2, mock_get.call_count)

    def test_download_if_url_passes_through_local_path(self):
        """A local path is returned unchanged and never triggers a download."""
        local = "resources/genome/betula/genome.gff.gz"

        with patch('fastdfe.io_handlers.requests.get') as mock_get:
            result = download_if_url(local)

        self.assertEqual(local, result)
        mock_get.assert_not_called()

    def test_download_if_url_downloads_remote(self):
        """A URL is downloaded and the local cache path is returned."""
        with patch('fastdfe.io_handlers.requests.get',
                   return_value=_fake_response(self.payload)) as mock_get:
            result = download_if_url(self.url)

        self.assertEqual(self.cached_path, result)
        mock_get.assert_called_once()
