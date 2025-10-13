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
        self.client.get_cached_image_result = MagicMock(return_value=None)
        self.client.list_schedulers = MagicMock(return_value=["default"])
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
            board_id="board-123",
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
            board_id="board-123",
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


    def test_browse_board_images_uses_cached_preview_when_available(self):
        board = {"name": "Board", "id": "board-1"}
        entry_one = {"image_name": "first.png"}
        entry_two = {"image_name": "second.png"}
        self.client.list_board_images.return_value = [entry_one, entry_two]

        retrieved_result = {
            "path": Path("/tmp/cache/first.png"),
            "metadata_path": Path("/tmp/cache/first.json"),
            "metadata": {"prompt": "one"},
            "cached": True,
        }
        cached_result = {
            "path": Path("/tmp/cache/second.png"),
            "metadata_path": Path("/tmp/cache/second.json"),
            "metadata": {"prompt": "two"},
            "cached": True,
        }

        def cache_lookup(name):
            return cached_result if name == "second.png" else None

        self.client.retrieve_board_image.return_value = retrieved_result
        self.client.get_cached_image_result.side_effect = cache_lookup

        with patch.object(TerminalAI, "clear_screen"), patch.object(
            TerminalAI, "_print_board_view_header"
        ), patch.object(
            TerminalAI, "_print_board_image_summary"
        ), patch.object(
            TerminalAI, "display_with_chafa"
        ), patch.object(
            TerminalAI, "get_input", return_value=""
        ), patch.object(
            TerminalAI, "get_key", side_effect=["RIGHT", "ESC"]
        ):
            TerminalAI._browse_board_images(self.client, board)

        self.client.retrieve_board_image.assert_called_once_with(
            image_info=entry_one, board_name="Board"
        )
        self.assertEqual(self.client.get_cached_image_result.call_count, 2)
        self.client.get_cached_image_result.assert_any_call("first.png")
        self.client.get_cached_image_result.assert_any_call("second.png")

    def test_poll_invoke_batch_status_returns_preview(self):
        preview = {"path": Path("/tmp/preview.png"), "metadata": {}}
        statuses = [
            {"status": "processing", "total": 2, "completed": 1, "processing": 1, "pending": 1},
            {
                "status": "completed",
                "total": 2,
                "completed": 2,
                "processing": 0,
                "pending": 0,
                "preview": preview,
            },
        ]
        self.client.get_batch_status.side_effect = statuses

        monotonic_values = [0, 0, 1, 1]

        with patch("builtins.print") as mock_print, patch.object(
            TerminalAI.time, "monotonic", side_effect=monotonic_values
        ), patch.object(TerminalAI.time, "sleep") as mock_sleep:
            status = TerminalAI._poll_invoke_batch_status(
                self.client, "batch-xyz", timeout=10, poll_interval=0
            )

        self.assertEqual(status, statuses[-1])
        self.assertEqual(self.client.get_batch_status.call_count, 2)
        self.client.get_batch_status.assert_called_with(
            "batch-xyz", include_preview=True, board_name=TerminalAI.TERMINALAI_BOARD_NAME
        )
        progress_calls = [call for call in mock_print.call_args_list if "Batch status:" in call.args[0]]
        self.assertGreaterEqual(len(progress_calls), 1)
        mock_sleep.assert_not_called()

    def test_poll_invoke_batch_status_handles_error(self):
        self.client.get_batch_status.side_effect = TerminalAI.InvokeAIClientError("boom")

        with patch("builtins.print") as mock_print, patch.object(
            TerminalAI.time, "monotonic", side_effect=[0, 0]
        ):
            status = TerminalAI._poll_invoke_batch_status(
                self.client, "batch-err", timeout=10, poll_interval=0
            )

        self.assertIsNone(status)
        self.assertTrue(
            any("Failed to poll batch" in call.args[0] for call in mock_print.call_args_list)
        )

    def test_run_generation_flow_polls_and_displays_preview(self):
        preview = {"path": Path("/tmp/preview.png"), "metadata": {}}
        polled_status = {"status": "completed", "preview": preview}

        with patch.object(TerminalAI, "select_invoke_model", side_effect=[self.model, None]), patch.object(
            TerminalAI, "clear_screen"
        ), patch.object(
            TerminalAI, "_print_invoke_prompt_header"
        ), patch.object(
            TerminalAI, "_invoke_generate_image", return_value={"queue_item_id": "q", "batch_id": "batch-1"}
        ), patch.object(
            TerminalAI, "_poll_invoke_batch_status", return_value=polled_status
        ) as poll_mock, patch.object(
            TerminalAI, "_present_invoke_preview", return_value=None
        ) as present_mock, patch.object(
            TerminalAI, "prompt_int", side_effect=[512, 512, 20]
        ), patch.object(
            TerminalAI, "prompt_float", return_value=7.5
        ), patch.object(
            TerminalAI, "select_scheduler_option", return_value="scheduler"
        ), patch.object(
            TerminalAI, "get_input", side_effect=["Prompt", "", "", "", "ESC"]
        ):
            TerminalAI._run_generation_flow(self.client, [self.model])

        poll_mock.assert_called_once()
        args, kwargs = poll_mock.call_args
        self.assertEqual(args, (self.client, "batch-1"))
        self.assertEqual(kwargs, {})
        present_mock.assert_called_once_with(preview)


if __name__ == "__main__":
    unittest.main()
