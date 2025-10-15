import base64
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from automatic1111_client import (  # noqa: E402
    Automatic1111Client,
    Automatic1111ClientError,
    Automatic1111Model,
)


class Automatic1111ClientTests(TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tempdir.name)
        self.client = Automatic1111Client(
            "1.2.3.4", 7860, nickname="demo", data_dir=self.data_dir
        )

    def tearDown(self) -> None:  # pragma: no cover - cleanup helper
        self.tempdir.cleanup()

    @patch("automatic1111_client.requests.get")
    def test_list_models_parses_entries(self, mock_get):
        payload = [
            {
                "model_name": "model.ckpt",
                "title": "Model Title",
                "hash": "abc123",
            },
            {"name": "other.safetensors", "title": "Other Model", "sha256": "def"},
        ]
        response = MagicMock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        models = self.client.list_models()

        mock_get.assert_called_once_with(
            "http://1.2.3.4:7860/sdapi/v1/sd-models", timeout=10
        )
        self.assertEqual(len(models), 2)
        self.assertIsInstance(models[0], Automatic1111Model)
        self.assertEqual(models[0].name, "model.ckpt")
        self.assertEqual(models[0].model_hash, "abc123")
        self.assertEqual(models[1].name, "other.safetensors")

    @patch("automatic1111_client.requests.post")
    def test_set_active_model_posts_checkpoint(self, mock_post):
        response = MagicMock()
        response.raise_for_status.return_value = None
        mock_post.return_value = response

        model = Automatic1111Model(
            name="model.ckpt",
            title="Model Title [hash]",
            model_hash="hash",
            raw={},
        )

        self.client.set_active_model(model)

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(
            kwargs["json"], {"sd_model_checkpoint": "Model Title [hash]"}
        )
        self.assertEqual(
            kwargs["timeout"], 15,
        )

    @patch("automatic1111_client.time.time", return_value=1700000000)
    @patch("automatic1111_client.requests.post")
    def test_txt2img_decodes_and_writes_image(self, mock_post, mock_time):
        image_bytes = b"fake-image"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        info_payload = {"width": 640, "height": 480, "sampler_name": "Euler a"}
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "images": [encoded],
            "parameters": {"Seed": 123, "Steps": 40, "CFG scale": 6.5},
            "info": json.dumps(info_payload),
        }
        mock_post.return_value = response

        model = Automatic1111Model(
            name="model.ckpt",
            title="Model Title",
            model_hash="abc123",
            raw={},
        )

        result = self.client.txt2img(
            model=model,
            prompt="Prompt",
            negative_prompt="",
            width=640,
            height=480,
            steps=30,
            cfg_scale=7.0,
            sampler="Euler a",
            seed=321,
        )

        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["timeout"], 60.0)
        self.assertEqual(kwargs["json"]["width"], 640)
        self.assertEqual(kwargs["json"]["height"], 480)
        self.assertEqual(kwargs["json"]["cfg_scale"], 7.0)
        self.assertEqual(kwargs["json"].get("sampler_name"), "Euler a")

        path = result["path"]
        self.assertTrue(path.exists())
        with path.open("rb") as handle:
            self.assertEqual(handle.read(), image_bytes)

        metadata_path = result["metadata_path"]
        self.assertTrue(metadata_path.exists())
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        self.assertEqual(metadata["seed"], 123)
        self.assertEqual(metadata["sampler"], "Euler a")
        self.assertEqual(metadata["width"], 640)
        self.assertEqual(metadata["height"], 480)
        self.assertEqual(metadata["server"]["ip"], "1.2.3.4")
        self.assertEqual(metadata["model"]["name"], "model.ckpt")

    @patch("automatic1111_client.requests.post")
    def test_txt2img_requires_image_payload(self, mock_post):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"images": []}
        mock_post.return_value = response

        model = Automatic1111Model(
            name="model.ckpt",
            title="Model Title",
            model_hash=None,
            raw={},
        )

        with self.assertRaises(Automatic1111ClientError):
            self.client.txt2img(
                model=model,
                prompt="Prompt",
                negative_prompt="",
                width=512,
                height=512,
                steps=30,
                cfg_scale=7.0,
                sampler="Euler a",
                seed=None,
            )
