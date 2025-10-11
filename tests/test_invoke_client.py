import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import call, patch

import requests

rain_stub = types.ModuleType("rain")
rain_stub.rain = lambda *args, **kwargs: None
sys.modules.setdefault("rain", rain_stub)

curses_nav_stub = types.ModuleType("curses_nav")
curses_nav_stub.get_input = lambda *args, **kwargs: ""
curses_nav_stub.interactive_menu = lambda *args, **kwargs: None
sys.modules.setdefault("curses_nav", curses_nav_stub)

sys.modules.setdefault("invoke_client", importlib.import_module("scripts.invoke_client"))

import scripts.TerminalAI
from scripts.invoke_client import (
    DEFAULT_SCHEDULER,
    InvokeAIClient,
    InvokeAIModel,
    InvokeAIClientError,
    QUEUE_ID,
)


class DummyResponse:
    def __init__(self, json_data=None, status_code=200, headers=None, content=b""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content

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

    def test_check_health_queue_failure_is_tolerated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = InvokeAIClient(TEST_SERVER_IP, 9090, data_dir=Path(tmpdir))
            version_response = DummyResponse({"version": "3.5.1"})
            queue_response = DummyResponse(status_code=503)

            with patch(
                "requests.get",
                side_effect=[
                    version_response,
                    queue_response,
                ],
            ) as mock_get:
                info = client.check_health()

        self.assertEqual(info["version"], "3.5.1")
        self.assertIsNone(info["queue_size"])
        expected_calls = [
            call(f"http://{TEST_SERVER_IP}:9090/api/v1/app/version", timeout=5),
            call(f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/size", timeout=5),
        ]
        self.assertEqual(mock_get.call_args_list, expected_calls)

    def test_check_invoke_api_returns_true_when_queue_unavailable(self):
        with patch("scripts.TerminalAI.InvokeAIClient") as mock_client_cls:
            client_instance = mock_client_cls.return_value
            client_instance.check_health.return_value = {"version": "3.5.1", "queue_size": None}

            result = scripts.TerminalAI.check_invoke_api(TEST_SERVER_IP, 9090)

        self.assertTrue(result)
        mock_client_cls.assert_called_once()
        client_instance.check_health.assert_called_once_with()
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


class InvokeSchedulerDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.client = InvokeAIClient(TEST_SERVER_IP, 9090, data_dir=Path(self.tmpdir.name))

    def test_list_schedulers_prefers_metadata(self):
        metadata_payload = {
            "metadata": {
                "system": {"schedulers": ["euler", "dpmpp_2m"]},
            }
        }

        def fake_get(url, *_, **__):
            if url.endswith("/api/v1/app/metadata"):
                return DummyResponse(metadata_payload)
            if url.endswith("/api/v1/metadata/schedulers"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/metadata"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/app/config"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/app/configuration"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/app/schedulers"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/schedulers"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/samplers"):
                return DummyResponse(status_code=404)
            raise AssertionError(f"Unexpected url: {url}")

        with patch("requests.get", side_effect=fake_get) as mock_get:
            schedulers = self.client.list_schedulers()

        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(schedulers, ["euler", "dpmpp_2m"])

    def test_list_schedulers_from_openapi(self):
        spec_payload = {
            "openapi": "3.1.0",
            "components": {
                "schemas": {
                    "SchedulerParam": {
                        "title": "Scheduler",
                        "type": "string",
                        "enum": ["ddim", "dpmpp_2m", "uni_pc"],
                    }
                }
            },
        }

        def fake_get(url, *_, **__):
            if url.endswith("/api/v1/app/metadata"):
                return DummyResponse(status_code=404)
            if url.endswith("/openapi.json"):
                return DummyResponse(spec_payload)
            if url.endswith("/api/v1/metadata/schedulers"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/metadata"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/app/config"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/app/configuration"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/app/schedulers"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/schedulers"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/samplers"):
                return DummyResponse(status_code=404)
            raise AssertionError(f"Unexpected url: {url}")

        with patch("requests.get", side_effect=fake_get) as mock_get:
            schedulers = self.client.list_schedulers()

        self.assertGreaterEqual(mock_get.call_count, 2)
        self.assertIn("ddim", schedulers)
        self.assertIn("dpmpp_2m", schedulers)
        self.assertIn("uni_pc", schedulers)

    def test_list_schedulers_falls_back_to_default(self):
        empty_spec = {"openapi": "3.1.0", "paths": {}}

        def fake_get(url, *_, **__):
            if url.endswith("/api/v1/app/metadata"):
                return DummyResponse(status_code=404)
            if url.endswith("/openapi.json"):
                return DummyResponse(empty_spec)
            if url.endswith("/api/v1/metadata/schedulers"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/metadata"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/app/config"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/app/configuration"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/app/schedulers"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/schedulers"):
                return DummyResponse(status_code=404)
            if url.endswith("/api/v1/samplers"):
                return DummyResponse(status_code=404)
            raise AssertionError(f"Unexpected url: {url}")

        with patch("requests.get", side_effect=fake_get):
            schedulers = self.client.list_schedulers()

        self.assertEqual(schedulers, [DEFAULT_SCHEDULER])

    def test_list_schedulers_raises_on_network_failure(self):
        with patch("requests.get", side_effect=requests.RequestException("boom")):
            with self.assertRaises(InvokeAIClientError):
                self.client.list_schedulers()


class InvokeQueueHandlingTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.client = InvokeAIClient(TEST_SERVER_IP, 9090, data_dir=Path(self.tmpdir.name))

    def tearDown(self):
        self.client = None

    def test_extract_queue_item_id_from_location_header(self):
        item_id = self.client._extract_queue_item_id_from_location(
            "http://server/api/v1/queue/default/i/12345"
        )
        self.assertEqual(item_id, "12345")

        self.assertIsNone(self.client._extract_queue_item_id_from_location(None))
        self.assertIsNone(self.client._extract_queue_item_id_from_location(""))

    def test_extract_queue_item_id_supports_camel_case_keys(self):
        payload = {
            "queueItemId": "abc-123",
            "nested": {"itemIds": ["should-not-be-picked"]},
        }

        self.assertEqual(self.client._extract_queue_item_id(payload), "abc-123")

    def test_generate_image_uses_session_when_queue_id_missing(self):
        model = InvokeAIModel(
            name="test-model",
            base="sdxl",
            key="model-key",
            raw={
                "key": "model-key",
                "hash": "hash",
                "name": "Test Model",
                "base": "sdxl",
                "type": "main",
            },
        )

        session_payload = {
            "id": "session-abc",
            "results": {
                "latents_to_image": {
                    "image": {
                        "image_name": "image.png",
                    }
                }
            },
        }

        enqueue_response = DummyResponse(
            {"session": session_payload},
            headers={"Location": "http://server/api/v1/queue/default/i/terminal_sdxl_graph"},
        )
        image_response = DummyResponse(status_code=200, content=b"fake-bytes")

        with patch("requests.post", return_value=enqueue_response) as mock_post, patch(
            "requests.get", return_value=image_response
        ) as mock_get, patch.object(self.client, "_poll_queue") as mock_poll:
            result = self.client.generate_image(
                model=model,
                prompt="a hamburger",
                negative_prompt="",
                width=512,
                height=512,
                steps=20,
                cfg_scale=7.5,
                scheduler=DEFAULT_SCHEDULER,
                seed=123,
                timeout=5,
            )

        mock_post.assert_called_once()
        mock_poll.assert_not_called()
        mock_get.assert_called_once()
        self.assertIsNone(result["metadata"]["queue_item"])
        self.assertEqual(result["metadata"]["session_id"], "session-abc")
        self.assertTrue(result["path"].exists())
        self.assertEqual(result["path"].read_bytes(), b"fake-bytes")

if __name__ == "__main__":
    unittest.main()
