import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import requests

from scripts.invoke_client import (
    InvokeAIClient,
    InvokeAIClientError,
    InvokeAIModel,
    QUEUE_ID,
)


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
class InvokeGraphBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.client = InvokeAIClient(TEST_SERVER_IP, 9090, data_dir=Path(self.tmpdir.name))

    def test_build_sd_graph_structure(self):
        model = InvokeAIModel(
            name="test-sd",
            base="sd-1",
            key="model-key",
            raw={
                "key": "model-key",
                "hash": "hash",
                "name": "Test Model",
                "base": "sd-1",
                "type": "main",
            },
        )

        info = self.client._build_graph(
            model=model,
            prompt="a sunrise over the ocean",
            negative_prompt="blurry",
            width=512,
            height=512,
            steps=20,
            cfg_scale=7.5,
            scheduler="dpmpp_2m",
            seed=123,
        )

        graph = info["graph"]
        self.assertIn("positive_conditioning", graph["nodes"])
        self.assertNotIn("pos_collect", graph["nodes"])
        self.assertEqual(graph["nodes"]["noise"]["seed"], 123)
        self.assertIn(
            {
                "source": {"node_id": "positive_conditioning", "field": "conditioning"},
                "destination": {"node_id": "denoise", "field": "positive_conditioning"},
            },
            graph["edges"],
        )
        self.assertEqual(info["output"], "latents_to_image")
        self.assertIsNone(info["data"])

    def test_build_sdxl_graph_structure(self):
        model = InvokeAIModel(
            name="test-sdxl",
            base="sdxl",
            key="sdxl-key",
            raw={
                "key": "sdxl-key",
                "hash": "hash",
                "name": "SDXL Model",
                "base": "sdxl",
                "type": "main",
            },
        )

        info = self.client._build_graph(
            model=model,
            prompt="futuristic city skyline",
            negative_prompt="low quality",
            width=1024,
            height=1024,
            steps=25,
            cfg_scale=6.5,
            scheduler="dpmpp_2m",
            seed=456,
        )

        graph = info["graph"]
        nodes = graph["nodes"]
        self.assertIn("positive_conditioning", nodes)
        self.assertEqual(nodes["positive_conditioning"]["target_width"], 1024)
        self.assertNotIn("pos_collect", nodes)
        self.assertIn(
            {
                "source": {"node_id": "model_loader", "field": "clip2"},
                "destination": {"node_id": "negative_conditioning", "field": "clip2"},
            },
            graph["edges"],
        )
        self.assertEqual(info["output"], "latents_to_image")
        self.assertIsNone(info["data"])


if __name__ == "__main__":
    unittest.main()
