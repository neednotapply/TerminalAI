import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import requests

from scripts.invoke_client import InvokeAIClient, InvokeAIClientError, QUEUE_ID


class DummyResponse:
    def __init__(self, json_data=None, status_code=200):
        self._json_data = json_data or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._json_data


TEST_SERVER_IP = "203.0.113.10"


class InvokeClientHealthTests(unittest.TestCase):
    def test_check_health_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = InvokeAIClient(TEST_SERVER_IP, 9090, data_dir=Path(tmpdir))
            version_response = DummyResponse({"version": "3.5.1"})
            queue_response = DummyResponse({"queue": {"size": 0}})

            with patch("requests.get", side_effect=[version_response, queue_response]) as mock_get:
                info = client.check_health()

        self.assertEqual(info["version"], "3.5.1")
        self.assertEqual(info["queue_size"], 0)
        expected_version_url = f"http://{TEST_SERVER_IP}:9090/api/v1/app/version"
        expected_queue_url = f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/size"
        self.assertEqual(
            mock_get.call_args_list,
            [
                call(expected_version_url, timeout=5),
                call(expected_queue_url, timeout=5),
            ],
        )

    def test_check_health_queue_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = InvokeAIClient(TEST_SERVER_IP, 9090, data_dir=Path(tmpdir))
            version_response = DummyResponse({"version": "3.5.1"})
            queue_response = DummyResponse(status_code=503)

            with patch(
                "requests.get",
                side_effect=[
                    version_response,
                    queue_response,
                    version_response,
                    queue_response,
                ],
            ) as mock_get:
                with self.assertRaises(InvokeAIClientError) as ctx:
                    client.check_health()

        self.assertIn("queue", str(ctx.exception).lower())
        expected_calls = [
            call(f"http://{TEST_SERVER_IP}:9090/api/v1/app/version", timeout=5),
            call(f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/size", timeout=5),
            call(f"https://{TEST_SERVER_IP}:9090/api/v1/app/version", timeout=5),
            call(f"https://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/size", timeout=5),
        ]
        self.assertEqual(mock_get.call_args_list, expected_calls)


if __name__ == "__main__":
    unittest.main()
