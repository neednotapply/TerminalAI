import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import TerminalAI  # noqa: E402
from automatic1111_client import Automatic1111Model  # noqa: E402


class TerminalAIAutomatic1111Tests(TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        self.client.txt2img = MagicMock()
        self.client.list_samplers = MagicMock(return_value=["Euler a", "UniPC"])
        self.client.set_active_model = MagicMock()
        self.model = Automatic1111Model(
            name="model.ckpt",
            title="Model Title",
            model_hash="hash",
            raw={},
        )

    def test_automatic1111_generate_image_forwards_arguments(self):
        self.client.txt2img.return_value = {"path": Path("/tmp/out.png")}

        result = TerminalAI._automatic1111_generate_image(
            self.client,
            self.model,
            "  prompt text  ",
            "  negative  ",
            513,
            511,
            28,
            6.5,
            "  sampler  ",
            None,
            45,
        )

        self.client.txt2img.assert_called_once_with(
            model=self.model,
            prompt="prompt text",
            negative_prompt="negative",
            width=512,
            height=504,
            steps=28,
            cfg_scale=6.5,
            sampler="sampler",
            seed=None,
            timeout=45,
        )
        self.assertEqual(result, {"path": Path("/tmp/out.png")})

    def test_automatic1111_generate_image_requires_prompt(self):
        with self.assertRaises(TerminalAI.Automatic1111ClientError):
            TerminalAI._automatic1111_generate_image(
                self.client,
                self.model,
                "   ",
                "",
                512,
                512,
                30,
                7.0,
                "",
                None,
                30,
            )

    def test_run_automatic1111_flow_generates_and_displays_image(self):
        self.client.txt2img.return_value = {"metadata": {"seed": 111, "sampler": "Sampler"}}

        with patch.object(TerminalAI, "select_automatic1111_model", side_effect=[self.model, None]), patch.object(
            TerminalAI, "prompt_int", side_effect=[513, 511, 20]
        ), patch.object(TerminalAI, "prompt_float", return_value=6.5), patch.object(
            TerminalAI, "select_sampler_option", return_value="  Sampler  "
        ), patch.object(
            TerminalAI, "get_input", side_effect=["  Prompt text  ", "  Negative  ", "", "ESC"]
        ), patch.object(TerminalAI, "clear_screen"), patch.object(
            TerminalAI, "_print_automatic1111_prompt_header"
        ), patch.object(
            TerminalAI, "_present_invoke_result"
        ) as present_mock:
            TerminalAI._run_automatic1111_flow(self.client, [self.model])

        self.client.set_active_model.assert_called_once_with(self.model)
        self.client.txt2img.assert_called_once_with(
            model=self.model,
            prompt="Prompt text",
            negative_prompt="Negative",
            width=512,
            height=504,
            steps=20,
            cfg_scale=6.5,
            sampler="Sampler",
            seed=None,
            timeout=TerminalAI.REQUEST_TIMEOUT,
        )
        present_mock.assert_called_once()
