import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from scripts.invoke_client import InvokeAIModel


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import TerminalAI  # noqa: E402


class TerminalAIInvokeTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.model = InvokeAIModel(name="model", base="sdxl", key=None, raw={})

    def test_invoke_generate_image_forwards_arguments(self):
        self.client.generate_image.return_value = {"path": "foo"}

        result = TerminalAI._invoke_generate_image(
            self.client,
            self.model,
            "  prompt text  ",
            "  negative  ",
            768,
            512,
            28,
            6.5,
            "  custom_scheduler  ",
            123,
            90,
        )

        self.client.generate_image.assert_called_once_with(
            model=self.model,
            prompt="prompt text",
            negative_prompt="negative",
            width=768,
            height=512,
            steps=28,
            cfg_scale=6.5,
            scheduler="custom_scheduler",
            seed=123,
            timeout=90.0,
        )
        self.assertEqual(result, {"path": "foo"})

    def test_invoke_generate_image_applies_defaults(self):
        self.client.generate_image.return_value = {"path": "bar"}

        result = TerminalAI._invoke_generate_image(
            self.client,
            self.model,
            "Prompt",
            None,
            640,
            640,
            30,
            7.5,
            "   ",
            None,
            45,
        )

        self.client.generate_image.assert_called_once_with(
            model=self.model,
            prompt="Prompt",
            negative_prompt="",
            width=640,
            height=640,
            steps=30,
            cfg_scale=7.5,
            scheduler=TerminalAI.DEFAULT_SCHEDULER,
            seed=None,
            timeout=45.0,
        )
        self.assertEqual(result, {"path": "bar"})

    def test_invoke_generate_image_requires_prompt(self):
        with self.assertRaises(TerminalAI.InvokeAIClientError):
            TerminalAI._invoke_generate_image(
                self.client,
                self.model,
                "   ",
                "",
                512,
                512,
                20,
                7.0,
                "",
                None,
                30,
            )


if __name__ == "__main__":
    unittest.main()
