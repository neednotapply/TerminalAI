import importlib
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
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
    UNCATEGORIZED_BOARD_ID,
)


class DummyResponse:
    def __init__(self, json_data=None, status_code=200, headers=None, content=b""):
        self._json_data = json_data or {}
        self.status_code = status_code
        self.headers = headers or {}
        if isinstance(content, str):
            self.content = content.encode()
            self._text = content
        elif isinstance(content, (bytes, bytearray)):
            self.content = bytes(content)
            self._text = self.content.decode(errors="ignore")
        else:
            text = str(content)
            self.content = text.encode()
            self._text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)

    def json(self):
        return self._json_data

    @property
    def text(self):
        return self._text


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
        self.assertIn("save_image", graph["nodes"])
        metadata = graph["nodes"]["save_image"].get("metadata")
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata.get("prompt"), "a sunrise over the ocean")
        self.assertEqual(metadata.get("negative_prompt"), "blurry")
        self.assertEqual(metadata.get("seed"), 123)
        self.assertEqual(metadata.get("width"), 512)
        self.assertEqual(metadata.get("height"), 512)
        self.assertEqual(metadata.get("steps"), 20)
        self.assertEqual(metadata.get("cfg_scale"), 7.5)
        self.assertEqual(metadata.get("scheduler"), "dpmpp_2m")
        model_metadata = metadata.get("model")
        self.assertEqual(model_metadata.get("key"), "model-key")
        self.assertEqual(model_metadata.get("hash"), "hash")
        self.assertEqual(model_metadata.get("name"), "Test Model")
        self.assertEqual(model_metadata.get("base"), "sd-1")
        self.assertEqual(model_metadata.get("type"), "main")
        self.assertEqual(info["output"], "save_image")
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
            width=1280,
            height=720,
            steps=25,
            cfg_scale=6.5,
            scheduler="dpmpp_2m",
            seed=456,
        )

        graph = info["graph"]
        nodes = graph["nodes"]
        self.assertIn("positive_conditioning", nodes)
        self.assertEqual(nodes["positive_conditioning"]["target_width"], 1280)
        self.assertNotIn("pos_collect", nodes)
        self.assertIn(
            {
                "source": {"node_id": "model_loader", "field": "clip2"},
                "destination": {"node_id": "negative_conditioning", "field": "clip2"},
            },
            graph["edges"],
        )
        self.assertIn("save_image", nodes)
        metadata = nodes["save_image"].get("metadata")
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata.get("prompt"), "futuristic city skyline")
        self.assertEqual(metadata.get("negative_prompt"), "low quality")
        self.assertEqual(metadata.get("seed"), 456)
        self.assertEqual(metadata.get("width"), 1280)
        self.assertEqual(metadata.get("height"), 720)
        self.assertEqual(metadata.get("steps"), 25)
        self.assertEqual(metadata.get("cfg_scale"), 6.5)
        self.assertEqual(metadata.get("scheduler"), "dpmpp_2m")
        model_metadata = metadata.get("model")
        self.assertEqual(model_metadata.get("key"), "sdxl-key")
        self.assertEqual(model_metadata.get("hash"), "hash")
        self.assertEqual(model_metadata.get("name"), "SDXL Model")
        self.assertEqual(model_metadata.get("base"), "sdxl")
        self.assertEqual(model_metadata.get("type"), "main")
        self.assertEqual(info["output"], "save_image")
        self.assertIsNone(info["data"])

    def test_build_graph_attaches_board_to_save_node(self):
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
            prompt="galaxy horizon",
            negative_prompt="",
            width=1280,
            height=720,
            steps=30,
            cfg_scale=7.0,
            scheduler="euler",
            seed=101,
            board_id="board-123",
            board_name="TerminalAI",
        )

        nodes = info["graph"]["nodes"]
        self.assertEqual(
            nodes["save_image"].get("board"),
            {"board_id": "board-123", "board_name": "TerminalAI"},
        )
        metadata = nodes["save_image"].get("metadata")
        self.assertEqual(metadata.get("prompt"), "galaxy horizon")
        self.assertEqual(metadata.get("negative_prompt"), "")
        self.assertEqual(metadata.get("seed"), 101)
        self.assertEqual(metadata.get("width"), 1280)
        self.assertEqual(metadata.get("height"), 720)
        self.assertEqual(metadata.get("steps"), 30)
        self.assertEqual(metadata.get("cfg_scale"), 7.0)
        self.assertEqual(metadata.get("scheduler"), "euler")
        self.assertIn(
            {
                "source": {"node_id": "latents_to_image", "field": "image"},
                "destination": {"node_id": "save_image", "field": "image"},
            },
            info["graph"]["edges"],
        )

    def test_build_flux_graph_structure(self):
        model = InvokeAIModel(
            name="flux-dev",
            base="flux",
            key=None,
            raw={"base": "flux", "type": "main", "name": "Flux.1 dev"},
        )

        info = self.client._build_graph(
            model=model,
            prompt="city skyline",
            negative_prompt="low quality",
            width=1024,
            height=768,
            steps=25,
            cfg_scale=3.5,
            scheduler="flux-default",
            seed=4242,
            board_id="board-flux",
            board_name="TerminalAI",
        )

        self.assertEqual(info["graph"]["id"], "terminal_flux_graph")
        nodes = info["graph"]["nodes"]
        self.assertEqual(nodes["model_loader"]["type"], "flux_model_loader")
        self.assertEqual(nodes["denoise"]["type"], "flux_denoise")
        self.assertEqual(nodes["vae_decode"]["type"], "flux_vae_decode")
        self.assertEqual(nodes["save_image"]["type"], "save_image")

        metadata = nodes["save_image"]["metadata"]
        self.assertEqual(metadata.get("prompt"), "city skyline")
        self.assertEqual(metadata.get("negative_prompt"), "low quality")
        self.assertEqual(metadata.get("seed"), 4242)
        self.assertEqual(metadata.get("width"), 1024)
        self.assertEqual(metadata.get("height"), 768)
        self.assertEqual(metadata.get("steps"), 25)
        self.assertEqual(metadata.get("cfg_scale"), 3.5)
        self.assertEqual(metadata.get("guidance"), 3.5)
        self.assertEqual(metadata.get("scheduler"), "flux-default")
        self.assertEqual(
            nodes["save_image"].get("board"),
            {"board_id": "board-flux", "board_name": "TerminalAI"},
        )

        self.assertIn("negative_conditioning", nodes)
        edges = info["graph"]["edges"]
        self.assertIn(
            {
                "source": {"node_id": "negative_conditioning", "field": "conditioning"},
                "destination": {
                    "node_id": "denoise",
                    "field": "negative_text_conditioning",
                },
            },
            edges,
        )
        self.assertIn(
            {
                "source": {"node_id": "vae_decode", "field": "image"},
                "destination": {"node_id": "save_image", "field": "image"},
            },
            edges,
        )

    def test_build_enqueue_payload_includes_board_name_with_id(self):
        model = InvokeAIModel(
            name="test-sd",
            base="sd-1",
            key=None,
            raw={"base": "sd-1", "type": "main"},
        )

        payload, graph_info, seed_value = self.client._build_enqueue_payload(
            model=model,
            prompt="forest", 
            negative_prompt="",
            width=512,
            height=512,
            steps=20,
            cfg_scale=7.5,
            scheduler="euler",
            seed=123,
            board_id="board-55",
            board_name="TerminalAI",
        )

        self.assertEqual(seed_value, 123)
        batch = payload["batch"]
        self.assertEqual(batch["board_id"], "board-55")
        self.assertEqual(batch["board_name"], "TerminalAI")
        save_node_board = graph_info["graph"]["nodes"]["save_image"].get("board")
        self.assertEqual(
            save_node_board,
            {"board_id": "board-55", "board_name": "TerminalAI"},
        )
        metadata = graph_info["graph"]["nodes"]["save_image"].get("metadata")
        self.assertEqual(metadata.get("prompt"), "forest")
        self.assertEqual(metadata.get("negative_prompt"), "")
        self.assertEqual(metadata.get("seed"), 123)
        self.assertEqual(metadata.get("width"), 512)
        self.assertEqual(metadata.get("height"), 512)
        self.assertEqual(metadata.get("steps"), 20)
        self.assertEqual(metadata.get("cfg_scale"), 7.5)
        self.assertEqual(metadata.get("scheduler"), "euler")


class InvokeClientBatchStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.client = InvokeAIClient(TEST_SERVER_IP, 9090, data_dir=Path(self.tmpdir.name))

    def test_get_batch_status_completed_with_preview(self):
        batch_id = "batch-123"
        status_url = (
            f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/b/{batch_id}/status"
        )
        list_url = f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/list_all"
        image_name = "image-abc.png"
        status_payload = {
            "total": 1,
            "completed": 1,
            "failed": 0,
            "pending": 0,
            "processing": 0,
            "status": "completed",
        }
        queue_entry = {
            "item_id": "item-789",
            "status": "completed",
            "batch_id": batch_id,
            "session": {
                "results": {
                    "save_image": {
                        "image": {
                            "image_name": image_name,
                            "metadata": {
                                "prompt": "test prompt",
                                "negative_prompt": "",
                                "width": 512,
                                "height": 512,
                                "steps": 20,
                                "cfg_scale": 7.5,
                                "seed": 42,
                                "scheduler": "euler",
                                "model": {"name": "model", "base": "sd-1"},
                            },
                        }
                    }
                },
                "source_prepared_mapping": {},
            },
        }
        image_content = b"fake-image-bytes"

        def fake_get(url, *args, **kwargs):
            if url == status_url:
                return DummyResponse(status_payload)
            if url == list_url:
                return DummyResponse([queue_entry])
            if url == f"http://{TEST_SERVER_IP}:9090/api/v1/images/i/{image_name}/full":
                return DummyResponse(content=image_content)
            raise AssertionError(f"Unexpected GET {url}")

        with patch("requests.get", side_effect=fake_get):
            status = self.client.get_batch_status(batch_id, include_preview=True)

        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["completed"], 1)
        self.assertEqual(status["queue_item_id"], "item-789")
        self.assertIn("preview", status)
        preview = status["preview"]
        self.assertTrue(Path(preview["path"]).exists())
        with Path(preview["path"]).open("rb") as fh:
            self.assertEqual(fh.read(), image_content)
        metadata = preview.get("metadata", {})
        self.assertEqual(metadata.get("prompt"), "test prompt")
        self.assertEqual(metadata.get("negative_prompt"), "")
        self.assertEqual(metadata.get("width"), 512)
        self.assertEqual(metadata.get("height"), 512)
        self.assertEqual(metadata.get("steps"), 20)
        self.assertEqual(metadata.get("cfg_scale"), 7.5)
        self.assertEqual(metadata.get("seed"), 42)
        self.assertEqual(metadata.get("scheduler"), "euler")
        self.assertEqual(metadata.get("model", {}).get("name"), "model")
        self.assertEqual(metadata.get("model", {}).get("base"), "sd-1")

    def test_get_batch_status_pending_without_preview(self):
        batch_id = "batch-456"
        status_url = (
            f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/b/{batch_id}/status"
        )
        list_url = f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/list_all"
        status_payload = {
            "total": 1,
            "completed": 0,
            "failed": 0,
            "pending": 1,
            "processing": 0,
            "status": "pending",
            "eta_seconds": "12.5",
        }
        queue_entry = {"item_id": "item-123", "status": "pending", "batch_id": batch_id}

        def fake_get(url, *args, **kwargs):
            if url == status_url:
                return DummyResponse(status_payload)
            if url == list_url:
                return DummyResponse([queue_entry])
            raise AssertionError(f"Unexpected GET {url}")

        with patch("requests.get", side_effect=fake_get):
            status = self.client.get_batch_status(batch_id, include_preview=True)

        self.assertEqual(status["status"], "pending")
        self.assertNotIn("preview", status)
        self.assertEqual(status.get("queue_item_id"), "item-123")
        self.assertEqual(status.get("eta_seconds"), 12.5)
        self.assertEqual(status.get("pending"), 1)
        self.assertIsNone(status.get("preview_error"))

    def test_get_batch_status_completed_uses_board_fallback(self):
        batch_id = "batch-fallback"
        status_url = (
            f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/b/{batch_id}/status"
        )
        list_url = f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/list_all"
        boards_url = f"http://{TEST_SERVER_IP}:9090/api/v1/boards/"
        images_url = f"http://{TEST_SERVER_IP}:9090/api/v1/images/"
        image_name = "fallback-image.png"

        status_payload = {
            "total": 1,
            "completed": 1,
            "failed": 0,
            "pending": 0,
            "processing": 0,
            "status": "completed",
        }
        queue_entry = {"item_id": "queue-1", "status": "completed", "batch_id": batch_id}
        board_entry = {"board_id": "board-1", "board_name": "TerminalAI"}
        board_image = {
            "image_name": image_name,
            "batch_id": batch_id,
            "metadata": {
                "prompt": "board prompt",
                "negative_prompt": "",
                "width": 512,
                "height": 512,
                "steps": 20,
                "cfg_scale": 7.5,
                "scheduler": "euler",
                "seed": 99,
                "model": {"name": "model", "base": "sd-1"},
                "batch_id": batch_id,
            },
        }
        image_content = b"fallback-image-content"

        def fake_get(url, *args, **kwargs):
            if url == status_url:
                return DummyResponse(status_payload)
            if url == list_url:
                return DummyResponse([queue_entry])
            if url == boards_url:
                params = kwargs.get("params")
                if params == {"all": True}:
                    return DummyResponse({"items": [board_entry]})
                raise AssertionError(f"Unexpected board params: {params}")
            if url == images_url:
                params = kwargs.get("params") or {}
                expected_params = {
                    "limit": 5,
                    "offset": 0,
                    "order_dir": "DESC",
                    "starred_first": "false",
                    "board_id": "board-1",
                }
                if params == expected_params:
                    return DummyResponse({"items": [board_image]})
                raise AssertionError(f"Unexpected image params: {params}")
            if url == f"http://{TEST_SERVER_IP}:9090/api/v1/images/i/{image_name}/full":
                return DummyResponse(content=image_content)
            raise AssertionError(f"Unexpected GET {url}")

        with patch("requests.get", side_effect=fake_get):
            status = self.client.get_batch_status(
                batch_id,
                include_preview=True,
                board_name="TerminalAI",
            )

        self.assertEqual(status["status"], "completed")
        self.assertEqual(status.get("queue_item_id"), "queue-1")
        self.assertIn("preview", status)
        preview = status["preview"]
        self.assertTrue(Path(preview["path"]).exists())
        with Path(preview["path"]).open("rb") as fh:
            self.assertEqual(fh.read(), image_content)
        metadata = preview.get("metadata", {})
        self.assertEqual(metadata.get("prompt"), "board prompt")
        self.assertEqual(metadata.get("image", {}).get("batch_id"), batch_id)
        self.assertEqual(metadata.get("seed"), 99)
        self.assertEqual(metadata.get("cfg_scale"), 7.5)
        self.assertEqual(metadata.get("scheduler"), "euler")

    def test_get_batch_status_missing_batch_raises(self):
        batch_id = "missing"
        status_url = (
            f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/b/{batch_id}/status"
        )

        with patch("requests.get", return_value=DummyResponse(status_code=404)) as mock_get:
            with self.assertRaises(InvokeAIClientError):
                self.client.get_batch_status(batch_id)

        mock_get.assert_called_once_with(status_url, timeout=10)


class InvokeSchedulerDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.client = InvokeAIClient(TEST_SERVER_IP, 9090, data_dir=Path(self.tmpdir.name))

    def test_ensure_board_returns_existing_id_on_422(self):
        error = requests.HTTPError(response=DummyResponse(status_code=422))
        with patch.object(
            self.client,
            "_fetch_board_id",
            side_effect=[None, "board-123"],
        ) as mock_fetch, patch(
            "requests.post",
            side_effect=error,
        ) as mock_post:
            board_id = self.client.ensure_board("TerminalAI")

        self.assertEqual(board_id, "board-123")
        self.assertEqual(
            mock_fetch.call_args_list,
            [call("TerminalAI"), call("TerminalAI")],
        )
        mock_post.assert_called_once_with(
            f"http://{TEST_SERVER_IP}:9090/api/v1/boards/",
            json={"name": "TerminalAI", "board_name": "TerminalAI"},
            timeout=15,
        )

    def test_ensure_board_sends_board_name(self):
        with patch.object(self.client, "_fetch_board_id", return_value=None) as mock_fetch, patch(
            "requests.post",
            return_value=DummyResponse({"id": "board-123"}),
        ) as mock_post:
            board_id = self.client.ensure_board("TerminalAI")

        self.assertEqual(board_id, "board-123")
        mock_fetch.assert_called_once_with("TerminalAI")
        mock_post.assert_called_once_with(
            f"http://{TEST_SERVER_IP}:9090/api/v1/boards/",
            json={"name": "TerminalAI", "board_name": "TerminalAI"},
            timeout=15,
        )

    def test_ensure_board_retries_with_query_parameter(self):
        error_payload = {
            "detail": [
                {
                    "type": "missing",
                    "loc": ["query", "board_name"],
                    "msg": "Field required",
                }
            ]
        }

        with patch.object(self.client, "_fetch_board_id", return_value=None) as mock_fetch, patch(
            "requests.post",
            side_effect=[
                requests.HTTPError(response=DummyResponse(error_payload, status_code=422)),
                DummyResponse({"id": "board-456"}),
            ],
        ) as mock_post:
            board_id = self.client.ensure_board("TerminalAI")

        self.assertEqual(board_id, "board-456")
        mock_fetch.assert_called_once_with("TerminalAI")
        self.assertEqual(
            mock_post.call_args_list,
            [
                call(
                    f"http://{TEST_SERVER_IP}:9090/api/v1/boards/",
                    json={"name": "TerminalAI", "board_name": "TerminalAI"},
                    timeout=15,
                ),
                call(
                    f"http://{TEST_SERVER_IP}:9090/api/v1/boards/",
                    params={"board_name": "TerminalAI"},
                    json={"name": "TerminalAI", "board_name": "TerminalAI"},
                    timeout=15,
                ),
            ],
        )

    def test_ensure_board_handles_numeric_ids(self):
        with patch.object(self.client, "_fetch_board_id", return_value=None) as mock_fetch, patch(
            "requests.post",
            return_value=DummyResponse({"id": 314}),
        ) as mock_post:
            board_id = self.client.ensure_board("TerminalAI")

        self.assertEqual(board_id, "314")
        mock_fetch.assert_called_once_with("TerminalAI")
        mock_post.assert_called_once_with(
            f"http://{TEST_SERVER_IP}:9090/api/v1/boards/",
            json={"name": "TerminalAI", "board_name": "TerminalAI"},
            timeout=15,
        )

    def test_fetch_board_id_recovers_from_missing_all_parameter(self):
        attempts: list[Optional[Dict[str, Any]]] = []

        def fake_get(url, params=None, timeout=15):
            self.assertEqual(url, f"http://{TEST_SERVER_IP}:9090/api/v1/boards/")
            attempts.append(params)
            if len(attempts) < 3:
                return DummyResponse(
                    {"detail": "Invalid request: Must provide either 'all' or both 'offset' and 'limit'"},
                    status_code=422,
                )
            return DummyResponse(
                {
                    "items": [
                        {
                            "board_id": "board-999",
                            "board_name": "TerminalAI",
                        }
                    ]
                }
            )

        with patch("requests.get", side_effect=fake_get) as mock_get:
            board_id = self.client._fetch_board_id("TerminalAI")

        self.assertEqual(board_id, "board-999")
        self.assertEqual(
            attempts,
            [
                {"all": True},
                {"all": "true"},
                {"offset": 0, "limit": 1000},
            ],
        )
        self.assertEqual(mock_get.call_count, 3)

    def test_fetch_board_id_allows_numeric_ids(self):
        payload = {"items": [{"board_id": 99, "board_name": "TerminalAI"}]}

        with patch.object(self.client, "_fetch_boards_payload", return_value=payload) as mock_payload:
            board_id = self.client._fetch_board_id("TerminalAI")

        self.assertEqual(board_id, "99")
        mock_payload.assert_called_once_with(strict=False)

    def test_list_boards_includes_numeric_ids(self):
        payload = [
            {"board_id": 77, "board_name": "TerminalAI", "image_count": 5},
            {"board_id": "UNASSIGNED", "board_name": "Uncategorized"},
        ]

        with patch.object(self.client, "_fetch_boards_payload", return_value=payload):
            boards = self.client.list_boards()

        board_names = {entry["name"]: entry for entry in boards}
        self.assertIn("TerminalAI", board_names)
        self.assertEqual(board_names["TerminalAI"]["id"], "77")
        self.assertEqual(board_names["TerminalAI"].get("count"), 5)
        self.assertEqual(board_names["Uncategorized"]["id"], UNCATEGORIZED_BOARD_ID)

    def test_normalize_board_id_maps_uncategorized_synonyms(self):
        self.assertEqual(self.client._normalize_board_id("none"), UNCATEGORIZED_BOARD_ID)
        self.assertEqual(self.client._normalize_board_id("UNASSIGNED"), UNCATEGORIZED_BOARD_ID)
        self.assertEqual(self.client._normalize_board_id(" unassigned "), UNCATEGORIZED_BOARD_ID)
        self.assertEqual(self.client._normalize_board_id("no-board"), UNCATEGORIZED_BOARD_ID)

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

    def test_poll_queue_by_batch_returns_session_and_item_id(self):
        session_payload = {"id": "session-123", "results": {}}
        status_payload = {"total": 1, "completed": 1}
        queue_payload = [
            {
                "item_id": 7,
                "status": "completed",
                "batch_id": "batch-xyz",
                "session": session_payload,
            }
        ]

        def fake_get(url, *_, **__):
            if url.endswith("/status"):
                return DummyResponse(status_payload)
            if url.endswith("/list_all"):
                return DummyResponse(queue_payload)
            raise AssertionError(f"Unexpected url: {url}")

        with patch("requests.get", side_effect=fake_get):
            result = self.client._poll_queue_by_batch(
                batch_id="batch-xyz",
                timeout=5,
                enqueue_started=time.time(),
            )

        self.assertIsNotNone(result)
        queue_item_id, session = result
        self.assertEqual(queue_item_id, "7")
        self.assertEqual(session, session_payload)


class InvokeSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.client = InvokeAIClient(TEST_SERVER_IP, 9090, data_dir=Path(self.tmpdir.name))

    def test_submit_image_generation_returns_identifiers(self):
        model = InvokeAIModel(name="demo", base="sdxl", key=None, raw={})
        payload = {"prepend": False, "batch": {}}
        graph_info = {"graph": {"id": "graph-1"}, "output": "out"}

        with patch.object(
            self.client,
            "_build_enqueue_payload",
            return_value=(payload, graph_info, 314),
        ) as mock_prepare, patch(
            "requests.post",
            return_value=DummyResponse(
                {"batch": {"batch_id": "batch-9"}},
                headers={"Location": f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/i/item-77"},
            ),
        ) as mock_post:
            result = self.client.submit_image_generation(
                model=model,
                prompt="sunrise",
                negative_prompt="",
                width=512,
                height=512,
                steps=25,
                cfg_scale=7.0,
                scheduler="euler",
                seed=123,
            )

        mock_prepare.assert_called_once_with(
            model=model,
            prompt="sunrise",
            negative_prompt="",
            width=512,
            height=512,
            steps=25,
            cfg_scale=7.0,
            scheduler="euler",
            seed=123,
            board_id=None,
            board_name=None,
        )
        mock_post.assert_called_once_with(
            f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/enqueue_batch",
            json=payload,
            timeout=15,
        )
        self.assertEqual(result["batch_id"], "batch-9")
        self.assertEqual(result["queue_item_id"], "item-77")
        self.assertEqual(result["seed"], 314)

    def test_submit_image_generation_ensures_board(self):
        model = InvokeAIModel(name="demo", base="sdxl", key=None, raw={})
        payload = {"prepend": False, "batch": {}}
        graph_info = {"graph": {"id": "graph-1"}, "output": "out"}

        with patch.object(self.client, "ensure_board", return_value="board-99") as mock_board, patch.object(
            self.client,
            "_build_enqueue_payload",
            return_value=(payload, graph_info, 101),
        ) as mock_prepare, patch(
            "requests.post",
            return_value=DummyResponse(
                {},
                headers={"Location": f"http://{TEST_SERVER_IP}:9090/api/v1/queue/{QUEUE_ID}/i/item-55"},
            ),
        ):
            self.client.submit_image_generation(
                model=model,
                prompt="nebula",
                negative_prompt="",
                width=512,
                height=512,
                steps=30,
                cfg_scale=7.5,
                scheduler="euler",
                seed=None,
                board_name="TerminalAI",
            )

        mock_board.assert_called_once_with("TerminalAI")
        mock_prepare.assert_called_once_with(
            model=model,
            prompt="nebula",
            negative_prompt="",
            width=512,
            height=512,
            steps=30,
            cfg_scale=7.5,
            scheduler="euler",
            seed=None,
            board_id="board-99",
            board_name="TerminalAI",
        )

    def test_submit_image_generation_uses_explicit_board_id(self):
        model = InvokeAIModel(name="demo", base="sdxl", key=None, raw={})
        payload = {"prepend": False, "batch": {}}
        graph_info = {"graph": {"id": "graph-1"}, "output": "out"}

        with patch.object(self.client, "ensure_board") as mock_board, patch.object(
            self.client,
            "_build_enqueue_payload",
            return_value=(payload, graph_info, 202),
        ) as mock_prepare, patch(
            "requests.post",
            return_value=DummyResponse({"batch": {"batch_id": "batch-7"}}),
        ):
            result = self.client.submit_image_generation(
                model=model,
                prompt="aurora",
                negative_prompt="",
                width=512,
                height=512,
                steps=40,
                cfg_scale=8.0,
                scheduler="euler",
                seed=99,
                board_name="TerminalAI",
                board_id="board-explicit",
            )

        mock_board.assert_not_called()
        mock_prepare.assert_called_once_with(
            model=model,
            prompt="aurora",
            negative_prompt="",
            width=512,
            height=512,
            steps=40,
            cfg_scale=8.0,
            scheduler="euler",
            seed=99,
            board_id="board-explicit",
            board_name="TerminalAI",
        )
        self.assertEqual(result["seed"], 202)

if __name__ == "__main__":
    unittest.main()
