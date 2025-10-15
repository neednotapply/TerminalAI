import pandas as pd
from types import SimpleNamespace
from unittest.mock import patch

from scripts import shodanscan


@patch("scripts.shodanscan.Automatic1111Client")
def test_check_automatic1111_api_collects_models(mock_client):
    model_a = SimpleNamespace(title="Model A", name="model_a")
    model_b = SimpleNamespace(title="", name="model_b")
    mock_client.return_value.list_models.return_value = [model_a, model_b]

    ok, reason, models = shodanscan.check_automatic1111_api("1.2.3.4", 7860)

    assert ok is True
    assert reason == ""
    assert models == ["Model A", "model_b"]

    mock_client.assert_called_once()
    args, kwargs = mock_client.call_args
    assert args == ("1.2.3.4", 7860)
    assert kwargs["scheme"] == "http"
    assert kwargs["nickname"] == "1.2.3.4"


def test_find_new_automatic1111_persists_models(tmp_path, monkeypatch):
    class DummyAPI:
        def search(self, query, limit):
            return {
                "matches": [
                    {
                        "ip_str": "5.6.7.8",
                        "port": None,
                        "timestamp": "2024-01-01T00:00:00Z",
                        "hostnames": ["diffusion.example.com"],
                        "org": "Stable Org",
                        "isp": "Example ISP",
                        "location": {
                            "city": "Artville",
                            "region_code": "AV",
                            "country_name": "Imaginary",
                            "latitude": 12.34,
                            "longitude": 56.78,
                        },
                    }
                ]
            }

    captured_calls = []

    def fake_check(ip, port, hostname=None):
        captured_calls.append((ip, port, hostname))
        return True, "", ["SDXL Base", "SDXL Refiner"]

    monkeypatch.setattr(shodanscan, "ping_time", lambda ip, port: 25.5)
    monkeypatch.setattr(shodanscan, "check_automatic1111_api", fake_check)

    output_path = tmp_path / "automatic1111.endpoints.csv"
    monkeypatch.setitem(shodanscan.CSV_PATHS, "automatic1111", output_path)

    df = shodanscan.ensure_columns(pd.DataFrame(columns=[]), "automatic1111")
    result = shodanscan.find_new(
        DummyAPI(),
        df,
        {"query": "title:Stable Diffusion", "api_type": "automatic1111", "default_port": 7860},
        limit=1,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["port"] == 7860
    assert bool(row["is_active"]) is True
    assert row["available_models"] == "SDXL Base;SDXL Refiner"

    assert captured_calls == [("5.6.7.8", 7860, "diffusion.example.com")]

    shodanscan.save_dataframe("automatic1111", result)
    contents = output_path.read_text(encoding="utf-8")
    assert "automatic1111" in contents
    assert "SDXL Base;SDXL Refiner" in contents
