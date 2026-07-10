import requests

import ollama_compat


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.payload = payload or {}

    def json(self):
        return self.payload


def endpoint():
    return {"ip": "127.0.0.1", "port": "11434", "scheme": "http"}


def test_tags_authentication_excludes_server(monkeypatch):
    ollama_compat.clear_probe_cache()
    monkeypatch.setattr(
        ollama_compat.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(401),
    )

    result = ollama_compat.probe_ollama_endpoint(endpoint())

    assert result.status == ollama_compat.REQUIRES_AUTH
    assert not result.accessible


def test_embedding_only_models_are_not_chat_choices(monkeypatch):
    ollama_compat.clear_probe_cache()
    monkeypatch.setattr(
        ollama_compat.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            payload={"models": [{"name": "nomic-embed-text:latest"}]}
        ),
    )

    result = ollama_compat.probe_ollama_endpoint(endpoint())

    assert result.status == ollama_compat.NO_CHAT_MODELS
    assert result.embedding_models == ["nomic-embed-text:latest"]


def test_nested_model_metadata_marks_custom_embedding_model(monkeypatch):
    ollama_compat.clear_probe_cache()
    monkeypatch.setattr(
        ollama_compat.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            payload={
                "models": [
                    {"name": "custom-model", "details": {"family": "bert-embedding"}}
                ]
            }
        ),
    )

    result = ollama_compat.probe_ollama_endpoint(endpoint())

    assert result.status == ollama_compat.NO_CHAT_MODELS
    assert result.embedding_models == ["custom-model"]


def test_local_model_can_pass_when_cloud_model_requires_auth(monkeypatch):
    ollama_compat.clear_probe_cache()

    def fake_get(*args, **kwargs):
        return FakeResponse(
            payload={
                "models": [
                    {"name": "provider:cloud"},
                    {"name": "llama3:latest"},
                ]
            }
        )

    def fake_post(url, **kwargs):
        model = kwargs["json"]["model"]
        if model == "provider:cloud":
            return FakeResponse(401)
        return FakeResponse(payload={"message": {"content": "OK"}})

    monkeypatch.setattr(ollama_compat.requests, "get", fake_get)
    monkeypatch.setattr(ollama_compat.requests, "post", fake_post)

    result = ollama_compat.probe_ollama_endpoint(endpoint())

    assert result.status == ollama_compat.ACCESSIBLE
    assert result.chat_models == ["llama3:latest"]


def test_probe_skips_405_until_compatible_route(monkeypatch):
    ollama_compat.clear_probe_cache()
    monkeypatch.setattr(
        ollama_compat.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(payload={"models": [{"name": "llama3"}]}),
    )

    def fake_post(url, **kwargs):
        if url.endswith("/v1/chat"):
            return FakeResponse(payload={"message": {"content": "OK"}})
        return FakeResponse(405)

    monkeypatch.setattr(ollama_compat.requests, "post", fake_post)

    result = ollama_compat.probe_ollama_endpoint(endpoint())

    assert result.status == ollama_compat.ACCESSIBLE


def test_auth_routes_with_unsupported_fallbacks_require_auth(monkeypatch):
    ollama_compat.clear_probe_cache()
    monkeypatch.setattr(
        ollama_compat.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(payload={"models": [{"name": "llama3"}]}),
    )
    monkeypatch.setattr(
        ollama_compat.requests,
        "post",
        lambda url, **kwargs: FakeResponse(401)
        if "/chat" in url
        else FakeResponse(405),
    )

    result = ollama_compat.probe_ollama_endpoint(endpoint())

    assert result.status == ollama_compat.REQUIRES_AUTH


def test_all_405_routes_are_unavailable(monkeypatch):
    ollama_compat.clear_probe_cache()
    monkeypatch.setattr(
        ollama_compat.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(payload={"models": [{"name": "llama3"}]}),
    )
    monkeypatch.setattr(
        ollama_compat.requests,
        "post",
        lambda *args, **kwargs: FakeResponse(405),
    )

    result = ollama_compat.probe_ollama_endpoint(endpoint())

    assert result.status == ollama_compat.UNAVAILABLE
    assert "supported chat route" in result.reason


def test_tags_timeout_is_unavailable(monkeypatch):
    ollama_compat.clear_probe_cache()

    def timeout(*args, **kwargs):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(ollama_compat.requests, "get", timeout)

    result = ollama_compat.probe_ollama_endpoint(endpoint())

    assert result.status == ollama_compat.UNAVAILABLE
