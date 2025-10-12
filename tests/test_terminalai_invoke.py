import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.invoke_client import InvokeAIModel


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import TerminalAI  # noqa: E402


class TerminalAIInvokeTests(unittest.TestCase):
    def setUp(self):
        self.client = MagicMock()
        self.client.submit_image_generation = MagicMock()
        self.client.ensure_board = MagicMock(return_value="board-123")
        self.client.list_board_images = MagicMock()
        self.client.retrieve_board_image = MagicMock()
        self.model = InvokeAIModel(name="model", base="sdxl", key=None, raw={})

    def test_invoke_generate_image_forwards_arguments(self):
        self.client.submit_image_generation.return_value = {"queue_item_id": "abc"}

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

        self.client.submit_image_generation.assert_called_once_with(
            model=self.model,
            prompt="prompt text",
            negative_prompt="negative",
            width=768,
            height=512,
            steps=28,
            cfg_scale=6.5,
            scheduler="custom_scheduler",
            seed=123,
            board_name=TerminalAI.TERMINALAI_BOARD_NAME,
        )
        self.client.ensure_board.assert_called_once_with(TerminalAI.TERMINALAI_BOARD_NAME)
        self.assertEqual(result, {"queue_item_id": "abc"})

    def test_invoke_generate_image_applies_defaults(self):
        self.client.submit_image_generation.return_value = {"queue_item_id": None}

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

        self.client.submit_image_generation.assert_called_once_with(
            model=self.model,
            prompt="Prompt",
            negative_prompt="",
            width=640,
            height=640,
            steps=30,
            cfg_scale=7.5,
            scheduler=TerminalAI.DEFAULT_SCHEDULER,
            seed=None,
            board_name=TerminalAI.TERMINALAI_BOARD_NAME,
        )
        self.client.ensure_board.assert_called_once_with(TerminalAI.TERMINALAI_BOARD_NAME)
        self.assertEqual(result, {"queue_item_id": None})

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

    def test_invoke_generate_image_requires_board_id(self):
        self.client.ensure_board.return_value = ""

        with self.assertRaises(TerminalAI.InvokeAIClientError):
            TerminalAI._invoke_generate_image(
                self.client,
                self.model,
                "Prompt",
                "",
                512,
                512,
                20,
                7.0,
                "",
                None,
                30,
            )

    def test_invoke_generate_image_rejects_uncategorized_board(self):
        self.client.ensure_board.return_value = TerminalAI.UNCATEGORIZED_BOARD_ID

        with self.assertRaises(TerminalAI.InvokeAIClientError):
            TerminalAI._invoke_generate_image(
                self.client,
                self.model,
                "Prompt",
                "",
                512,
                512,
                20,
                7.0,
                "",
                None,
                30,
            )

    def test_browse_board_images_rejects_terminalai_uncategorized_board(self):
        board = {
            "name": TerminalAI.TERMINALAI_BOARD_NAME,
            "id": TerminalAI.UNCATEGORIZED_BOARD_ID,
        }

        with patch.object(TerminalAI, "get_input", return_value=""):
            TerminalAI._browse_board_images(self.client, board)

        self.client.list_board_images.assert_not_called()
        self.client.retrieve_board_image.assert_not_called()


if __name__ == "__main__":
    unittest.main()
