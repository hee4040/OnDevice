from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

from huggingface_hub.errors import LocalEntryNotFoundError
from requests import Response
from requests.exceptions import ConnectionError, HTTPError, ReadTimeout, SSLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_imagenet import retry_download


class DownloadRetryTests(unittest.TestCase):
    def test_wrapped_metadata_timeout_recovers_without_logging_signed_url(self):
        error = LocalEntryNotFoundError("file not in cache")
        error.__cause__ = ReadTimeout("https://example.com/?Signature=secret")
        operation = Mock(side_effect=[error, "/cache/shard.parquet"])
        output = io.StringIO()
        with patch("prepare_imagenet.time.sleep") as sleep, redirect_stdout(output):
            self.assertEqual(retry_download(operation, "shard", retries=2), "/cache/shard.parquet")
        self.assertEqual(operation.call_count, 2)
        sleep.assert_called_once_with(5)
        self.assertNotIn("Signature", output.getvalue())

    def test_permanent_http_errors_are_not_retried(self):
        for status in (401, 403, 404):
            with self.subTest(status=status):
                response = Response()
                response.status_code = status
                error = HTTPError(response=response)
                operation = Mock(side_effect=error)
                with patch("prepare_imagenet.time.sleep") as sleep:
                    with self.assertRaises(HTTPError):
                        retry_download(operation, "shard")
                self.assertEqual(operation.call_count, 1)
                sleep.assert_not_called()

    def test_rate_limit_and_wrapped_server_error_recover(self):
        for status in (429, 503):
            with self.subTest(status=status):
                response = Response()
                response.status_code = status
                error = LocalEntryNotFoundError("metadata unavailable")
                error.__cause__ = HTTPError(response=response)
                operation = Mock(side_effect=[error, "ok"])
                with patch("prepare_imagenet.time.sleep"), redirect_stdout(io.StringIO()):
                    self.assertEqual(retry_download(operation, "metadata"), "ok")
                self.assertEqual(operation.call_count, 2)

    def test_exhaustion_preserves_error_and_caps_backoff(self):
        error = ConnectionError("connection lost")
        operation = Mock(side_effect=error)
        output = io.StringIO()
        with patch("prepare_imagenet.time.sleep") as sleep, redirect_stdout(output):
            with self.assertRaises(ConnectionError) as raised:
                retry_download(operation, "shard", retries=6)
        self.assertIs(raised.exception, error)
        self.assertEqual(operation.call_count, 7)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10, 20, 40, 60, 60])
        self.assertIn("Run the same command again", output.getvalue())

    def test_certificate_and_local_errors_are_not_retried(self):
        for error in (SSLError("bad certificate"), ValueError("bad data"), OSError("disk full")):
            with self.subTest(error=type(error).__name__):
                operation = Mock(side_effect=error)
                with patch("prepare_imagenet.time.sleep") as sleep:
                    with self.assertRaises(type(error)):
                        retry_download(operation, "shard")
                sleep.assert_not_called()
                self.assertEqual(operation.call_count, 1)

    def test_zero_retries_makes_one_attempt(self):
        operation = Mock(side_effect=ReadTimeout("timeout"))
        with patch("prepare_imagenet.time.sleep") as sleep, redirect_stdout(io.StringIO()):
            with self.assertRaises(ReadTimeout):
                retry_download(operation, "shard", retries=0)
        self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
