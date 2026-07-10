"""Shared Ollama discovery and chat-compatibility checks."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


ACCESSIBLE = "accessible"
REQUIRES_AUTH = "requires_auth"
NO_CHAT_MODELS = "no_chat_models"
UNAVAILABLE = "unavailable"

_CACHE_TTL_SECONDS = 600
_DEFAULT_TIMEOUT = 5
_MINIMAL_PROMPT = "Reply with exactly OK."
_EMBEDDING_TOKENS = ("embed", "embedding", "rerank", "reranker")
_CACHE: Dict[Tuple[str, str], Tuple[float, "OllamaProbeResult"]] = {}
_CACHE_LOCK = Lock()


@dataclass
class OllamaProbeResult:
    status: str
    base_url: str = ""
    models: List[str] = field(default_factory=list)
    chat_models: List[str] = field(default_factory=list)
    embedding_models: List[str] = field(default_factory=list)
    reason: str = ""
    checked_models: List[str] = field(default_factory=list)

    @property
    def accessible(self) -> bool:
        return self.status == ACCESSIBLE


def _first_value(raw: Any) -> str:
    if raw is None:
        return ""
    value = str(raw).strip()
    for separator in (";", ",", " "):
        if separator in value:
            return value.split(separator, 1)[0].strip()
    return value


def endpoint_base_url(endpoint: dict) -> str:
    host = _first_value(
        endpoint.get("api_host") or endpoint.get("ip") or endpoint.get("hostnames")
    )
    port = _first_value(endpoint.get("port"))
    if not host or not port:
        return ""
    scheme = str(endpoint.get("scheme") or "http").strip().lower()
    if scheme not in {"http", "https"}:
        scheme = "http"
    return f"{scheme}://{host}:{port}".rstrip("/")


def endpoint_headers(endpoint: dict) -> Dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    token = endpoint.get("api_key") or endpoint.get("token") or endpoint.get("auth_token")
    if token:
        value = str(token)
        headers["Authorization"] = value if value.lower().startswith("bearer ") else f"Bearer {value}"
    return headers


def _parse_model_names(raw: Any) -> List[str]:
    if isinstance(raw, str):
        values: Iterable[Any] = raw.replace("|", ";").replace(",", ";").split(";")
    elif isinstance(raw, list):
        values = raw
    else:
        values = []
    names: List[str] = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name") or value.get("model") or value.get("id")
        if value and str(value).strip() and str(value).strip() not in names:
            names.append(str(value).strip())
    return names


def _is_embedding_model(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _EMBEDDING_TOKENS)


def _extract_models(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return []
    raw_models = payload.get("models") or payload.get("data") or []
    return _parse_model_names(raw_models)


def _metadata_contains_embedding(value: Any, depth: int = 0) -> bool:
    if depth > 4:
        return False
    if isinstance(value, str):
        lowered = value.lower()
        return any(token in lowered for token in _EMBEDDING_TOKENS)
    if isinstance(value, dict):
        return any(_metadata_contains_embedding(item, depth + 1) for item in value.values())
    if isinstance(value, list):
        return any(_metadata_contains_embedding(item, depth + 1) for item in value)
    return False


def _extract_embedding_models(payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return []
    raw_models = payload.get("models") or payload.get("data") or []
    names: List[str] = []
    for item in raw_models if isinstance(raw_models, list) else []:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model") or item.get("id")
        metadata = {
            key: item.get(key)
            for key in ("details", "model_info", "capabilities", "type", "family", "families")
            if key in item
        }
        if name and _metadata_contains_embedding(metadata):
            names.append(str(name).strip())
    return names


def _extract_content(payload: Any) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    message = payload.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        if message["content"].strip():
            return message["content"].strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        message = choice.get("message") or choice.get("delta")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            if message["content"].strip():
                return message["content"].strip()
    for key in ("response", "content", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _candidate_models(
    models: List[str], maximum: int, metadata_embedding: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    metadata_embedding_set = set(metadata_embedding or [])
    embedding = [
        model
        for model in models
        if model in metadata_embedding_set or _is_embedding_model(model)
    ]
    chat = [
        model
        for model in models
        if model not in metadata_embedding_set and not _is_embedding_model(model)
    ]
    # Prefer local models. Cloud models are still tested when they are the only
    # choices, but they must not displace likely public local models.
    chat.sort(key=lambda model: (":cloud" in model.lower() or "cloud" in model.lower(), models.index(model)))
    return chat[:maximum], embedding


def _chat_request(path: str, base_url: str, model: str, headers: Dict[str, str], timeout: float) -> requests.Response:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _MINIMAL_PROMPT}],
        "stream": False,
        "options": {"num_predict": 1},
    }
    if path.startswith("/v1/"):
        payload["max_tokens"] = 1
    return requests.post(
        f"{base_url}{path}", headers=headers, json=payload, timeout=timeout
    )


def _generate_request(path: str, base_url: str, model: str, headers: Dict[str, str], timeout: float) -> requests.Response:
    return requests.post(
        f"{base_url}{path}",
        headers=headers,
        json={
            "model": model,
            "prompt": _MINIMAL_PROMPT,
            "stream": False,
            "options": {"num_predict": 1},
        },
        timeout=timeout,
    )


def _probe_model(base_url: str, model: str, headers: Dict[str, str], timeout: float) -> Tuple[str, str]:
    auth_failures = 0
    route_failures = 0
    request_errors = 0
    model_rejections = 0
    for path in ("/api/chat", "/v1/chat/completions", "/v1/chat", "/chat"):
        try:
            response = _chat_request(path, base_url, model, headers, timeout)
        except requests.RequestException:
            request_errors += 1
            continue
        if response.status_code in (401, 403):
            auth_failures += 1
            continue
        if response.status_code in (404, 405):
            route_failures += 1
            continue
        if 200 <= response.status_code < 300:
            try:
                if _extract_content(response.json()):
                    return "ok", ""
            except ValueError:
                pass
            model_rejections += 1
            continue
        # 400/422 means the route answered but this model/API shape was not
        # accepted; continue to another compatible route/model.
        model_rejections += 1

    for path in ("/api/generate", "/generate"):
        try:
            response = _generate_request(path, base_url, model, headers, timeout)
        except requests.RequestException:
            request_errors += 1
            continue
        if response.status_code in (401, 403):
            auth_failures += 1
            continue
        if response.status_code in (404, 405):
            route_failures += 1
            continue
        if 200 <= response.status_code < 300:
            try:
                if _extract_content(response.json()):
                    return "ok", ""
            except ValueError:
                pass
            model_rejections += 1
            continue
        model_rejections += 1

    if auth_failures and not request_errors and not model_rejections:
        return REQUIRES_AUTH, "all chat routes require authentication"
    if request_errors and not auth_failures and not route_failures:
        return UNAVAILABLE, "chat request failed or timed out"
    if route_failures and not auth_failures:
        return UNAVAILABLE, "no supported chat route responded"
    return NO_CHAT_MODELS, f"model '{model}' was not chat-compatible"


def probe_ollama_endpoint(endpoint: dict, *, force: bool = False, max_models: int = 3, timeout: float = _DEFAULT_TIMEOUT) -> OllamaProbeResult:
    base_url = endpoint_base_url(endpoint)
    if not base_url:
        return OllamaProbeResult(UNAVAILABLE, reason="missing Ollama connection details")
    cache_key = (base_url, str(bool(endpoint_headers(endpoint).get("Authorization"))))
    now = time.time()
    if not force:
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            return cached[1]

    headers = endpoint_headers(endpoint)
    try:
        response = requests.get(f"{base_url}/api/tags", headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        result = OllamaProbeResult(UNAVAILABLE, base_url, reason=str(exc))
        _cache_result(cache_key, result)
        return result
    if response.status_code in (401, 403):
        result = OllamaProbeResult(REQUIRES_AUTH, base_url, reason="Ollama model discovery requires authentication")
        _cache_result(cache_key, result)
        return result
    if response.status_code >= 400:
        result = OllamaProbeResult(UNAVAILABLE, base_url, reason=f"Ollama model discovery returned HTTP {response.status_code}")
        _cache_result(cache_key, result)
        return result
    try:
        payload = response.json()
        models = _extract_models(payload)
        metadata_embedding = _extract_embedding_models(payload)
    except ValueError:
        models = []
        metadata_embedding = []
    if not models:
        result = OllamaProbeResult(NO_CHAT_MODELS, base_url, reason="Ollama returned no models")
        _cache_result(cache_key, result)
        return result

    candidates, embedding_models = _candidate_models(
        models, max_models, metadata_embedding
    )
    if not candidates:
        result = OllamaProbeResult(NO_CHAT_MODELS, base_url, models, [], embedding_models, "only embedding models were advertised")
        _cache_result(cache_key, result)
        return result

    successes: List[str] = []
    statuses: List[str] = []
    reasons: List[str] = []
    for model in candidates:
        status, reason = _probe_model(base_url, model, headers, timeout)
        statuses.append(status)
        reasons.append(reason)
        if status == "ok":
            successes.append(model)

    if successes:
        result = OllamaProbeResult(ACCESSIBLE, base_url, models, successes, embedding_models, checked_models=candidates)
    elif statuses and all(status == REQUIRES_AUTH for status in statuses):
        result = OllamaProbeResult(REQUIRES_AUTH, base_url, models, [], embedding_models, "all tested models require authentication", candidates)
    elif statuses and all(status == UNAVAILABLE for status in statuses):
        result = OllamaProbeResult(UNAVAILABLE, base_url, models, [], embedding_models, reasons[-1], candidates)
    else:
        result = OllamaProbeResult(NO_CHAT_MODELS, base_url, models, [], embedding_models, reasons[-1], candidates)
    _cache_result(cache_key, result)
    return result


def _cache_result(key: Tuple[str, str], result: OllamaProbeResult) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = (time.time(), result)


def clear_probe_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()
