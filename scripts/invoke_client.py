"""Helper utilities for interacting with InvokeAI servers."""
from __future__ import annotations

import datetime
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_SCHEDULER = "euler"
DEFAULT_SCHEDULER_OPTIONS: Tuple[str, ...] = (
    "ddim",
    "ddpm",
    "dpmpp_2m",
    "dpmpp_2m_karras",
    "dpmpp_2s",
    "dpmpp_sde",
    "dpmpp_sde_karras",
    "dpm_solver",
    "euler",
    "euler_a",
    "heun",
    "lms",
    "pndm",
    "uni_pc",
)
DEFAULT_CFG_SCALE = 7.5
DEFAULT_STEPS = 30
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
QUEUE_ID = "default"
ORIGIN = "terminalai"
DESTINATION = "terminalai"

UNCATEGORIZED_BOARD_ID = "UNASSIGNED"
"""Synthetic board id used for InvokeAI's Uncategorized images."""

UNCATEGORIZED_BOARD_NAME = "Uncategorized"
"""Human readable name for InvokeAI's Uncategorized images."""


def _build_static_model_endpoints() -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    """Return a deduplicated list of known InvokeAI model endpoints."""

    base_paths = [
        "/api/v2/models",
        "/api/v1/models",
        "/api/v1/models/main",
        "/api/v1/model-manager/models",
        "/api/v1/model-manager/models/main",
        "/api/v1/model_manager/models",
        "/api/v1/model_manager/models/main",
    ]

    param_variants: Tuple[Optional[Dict[str, Any]], ...] = (
        None,
        {"model_type": "main"},
        {"model_type": "primary"},
        {"model_type": "checkpoint"},
        {"model_type": "ckpt"},
        {"type": "main"},
        {"type": "primary"},
        {"type": "checkpoint"},
        {"type": "ckpt"},
    )

    endpoints: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    seen: set[Tuple[str, Tuple[Tuple[str, Any], ...]]] = set()

    for raw_path in base_paths:
        normalized = f"/{raw_path.strip('/')}" if raw_path else "/api/v1/models"
        path_variants = {normalized.rstrip("/")}
        path_variants.add(f"{normalized.rstrip('/')}/")

        for path in path_variants:
            for params in param_variants:
                param_items = tuple(sorted(params.items())) if params else tuple()
                key = (path, param_items)
                if key in seen:
                    continue
                seen.add(key)
                endpoints.append((path, params))

    return endpoints


STATIC_MODEL_ENDPOINTS: List[Tuple[str, Optional[Dict[str, Any]]]] = _build_static_model_endpoints()


class InvokeAIClientError(RuntimeError):
    """Raised when the InvokeAI client encounters an unrecoverable error."""


@dataclass
class InvokeAIModel:
    name: str
    base: str
    key: Optional[str]
    raw: Dict[str, Any]


class InvokeAIClient:
    """Minimal client for interacting with InvokeAI's invocation API."""

    def __init__(
        self,
        ip: str,
        port: int,
        nickname: Optional[str] = None,
        data_dir: Optional[Path] = None,
        scheme: str = "http",
        allow_https_fallback: bool = True,
    ) -> None:
        self.ip = ip
        self.port = port
        self.nickname = nickname or ip
        self._base_urls = self._build_base_urls(scheme, allow_https_fallback)
        self.base_url = self._base_urls[0]
        base_data = data_dir if data_dir is not None else Path(__file__).resolve().parent.parent / "data"
        self.images_dir = Path(base_data) / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check_health(self) -> Dict[str, Any]:
        """Ensure the InvokeAI invocation API is reachable."""

        last_error: Optional[requests.RequestException] = None
        version_payload: Any = None
        queue_payload: Any = None

        for base_url in list(self._base_urls):
            version_url = f"{base_url}/api/v1/app/version"
            queue_url = f"{base_url}/api/v1/queue/{QUEUE_ID}/size"

            try:
                version_resp = requests.get(version_url, timeout=5)
                version_resp.raise_for_status()
            except requests.RequestException as exc:
                last_error = exc
                continue

            version_payload = self._safe_json(version_resp)

            try:
                queue_resp = requests.get(queue_url, timeout=5)
                queue_resp.raise_for_status()
            except requests.RequestException:
                queue_payload = None
            else:
                queue_payload = self._safe_json(queue_resp)

            # Promote the successful base URL to the front of the list so future
            # requests prefer the working scheme.
            self._promote_base_url(base_url)
            break
        else:
            raise InvokeAIClientError("Failed to query InvokeAI version endpoint") from last_error

        info: Dict[str, Any] = {
            "version": None,
            "queue_size": None,
        }
        if isinstance(version_payload, dict):
            if "version" in version_payload:
                info["version"] = version_payload.get("version")
            elif "app_version" in version_payload:
                info["version"] = version_payload.get("app_version")
        if isinstance(queue_payload, dict):
            queue_data = queue_payload.get("queue") if isinstance(queue_payload.get("queue"), dict) else queue_payload
            if isinstance(queue_data, dict):
                if "size" in queue_data:
                    info["queue_size"] = queue_data.get("size")
                elif "queue_size" in queue_data:
                    info["queue_size"] = queue_data.get("queue_size")

        return info

    def list_models(self) -> List[InvokeAIModel]:
        """Return the available main models on the server."""

        soft_error_statuses = {400, 401, 403, 404, 405, 406, 415, 422, 429, 500, 503}
        last_error: Optional[BaseException] = None
        attempted: set[Tuple[str, str, Tuple[Tuple[str, Any], ...]]] = set()

        for base_url in list(self._base_urls):
            for path, params in self._candidate_model_endpoints(base_url):
                normalized_path = path if path.startswith("/") else f"/{path}"
                param_items = tuple(sorted(params.items())) if params else tuple()
                key = (base_url, normalized_path, param_items)
                if key in attempted:
                    continue
                attempted.add(key)

                url = f"{base_url}{normalized_path}"
                try:
                    resp = requests.get(url, params=params, timeout=10)
                    resp.raise_for_status()
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else None
                    if status in soft_error_statuses:
                        last_error = exc
                        continue
                    raise
                except requests.RequestException as exc:
                    last_error = exc
                    continue

                models = self._parse_models_payload(self._safe_json(resp))
                if not models:
                    continue

                models.sort(key=lambda m: m.name.lower())
                self._promote_base_url(base_url)
                return models

        if last_error is not None:
            raise InvokeAIClientError(
                "InvokeAI server did not expose a compatible models endpoint"
            ) from last_error

        raise InvokeAIClientError("InvokeAI server did not return a valid models list")

    def list_schedulers(self) -> List[str]:
        """Return the available scheduler names advertised by the server."""

        last_error: Optional[BaseException] = None
        for base_url in list(self._base_urls):
            try:
                installed = self._load_installed_schedulers(base_url)
            except requests.RequestException as exc:
                last_error = exc
                installed = []

            if installed:
                self._promote_base_url(base_url)
                return installed

            try:
                discovered = self._discover_scheduler_options(base_url)
            except requests.RequestException as exc:
                last_error = exc
                continue

            if discovered:
                self._promote_base_url(base_url)
                return discovered

        if last_error is not None:
            raise InvokeAIClientError(
                "Failed to query InvokeAI scheduler metadata"
            ) from last_error

        return [DEFAULT_SCHEDULER]

    def _safe_json(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return None

    def _candidate_model_endpoints(
        self, base_url: str
    ) -> Iterable[Tuple[str, Optional[Dict[str, Any]]]]:
        """Yield possible models endpoints, combining static and discovered paths."""

        for path, params in STATIC_MODEL_ENDPOINTS:
            yield path, params

        for path, params in self._discover_model_endpoints(base_url):
            yield path, params

    def _discover_model_endpoints(
        self, base_url: str
    ) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """Discover InvokeAI model endpoints from the server's OpenAPI schema."""

        try:
            spec = self._load_openapi_spec(base_url)
        except requests.RequestException:
            return []
        if not isinstance(spec, dict):
            return []

        discovered: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        seen: set[Tuple[str, Tuple[Tuple[str, Any], ...]]] = set()
        paths = spec.get("paths")
        if not isinstance(paths, dict):
            return []

        for raw_path, operations in paths.items():
            if not isinstance(raw_path, str):
                continue
            path_lower = raw_path.lower()
            if "model" not in path_lower:
                continue
            if "{" in raw_path:
                continue
            if not raw_path.startswith("/"):
                raw_path = f"/{raw_path}"

            get_op = operations.get("get") if isinstance(operations, dict) else None
            if not isinstance(get_op, dict):
                continue

            param_candidates: List[Optional[Dict[str, Any]]] = [None]
            parameters: List[Dict[str, Any]] = []
            for source in (operations.get("parameters"), get_op.get("parameters")):
                if isinstance(source, list):
                    parameters.extend(
                        [p for p in source if isinstance(p, dict) and p.get("in") == "query"]
                    )

            for param in parameters:
                name = param.get("name")
                if name not in {"model_type", "type"}:
                    continue
                schema = param.get("schema") if isinstance(param.get("schema"), dict) else {}
                values = schema.get("enum") if isinstance(schema.get("enum"), list) else []
                options = [v for v in values if isinstance(v, str)]
                if not options:
                    options = ["main", "primary", "checkpoint", "ckpt"]
                for value in options:
                    if value and value.lower() in {"main", "checkpoint", "primary", "ckpt"}:
                        param_candidates.append({name: value})

            for param in param_candidates:
                normalized_path = raw_path
                if normalized_path.endswith("//"):
                    normalized_path = normalized_path.rstrip("/")
                variants = {normalized_path}
                if normalized_path.endswith("/"):
                    variants.add(normalized_path.rstrip("/"))
                else:
                    variants.add(f"{normalized_path}/")
                for variant in variants:
                    param_items = tuple(sorted(param.items())) if param else tuple()
                    key = (variant, param_items)
                    if key in seen:
                        continue
                    seen.add(key)
                    discovered.append((variant, param))

        return discovered

    def _load_openapi_spec(self, base_url: str) -> Optional[Dict[str, Any]]:
        last_error: Optional[requests.RequestException] = None
        for suffix in ("/openapi.json", "/docs/openapi.json"):
            url = f"{base_url}{suffix}"
            try:
                resp = requests.get(url, timeout=5)
            except requests.RequestException as exc:
                last_error = exc
                continue

            if resp.status_code and resp.status_code >= 400:
                continue

            spec = self._safe_json(resp)
            if isinstance(spec, dict):
                return spec

        if last_error is not None:
            raise last_error

        return None

    def _discover_scheduler_options(self, base_url: str) -> List[str]:
        try:
            spec = self._load_openapi_spec(base_url)
        except requests.RequestException as exc:
            raise exc
        if not isinstance(spec, dict):
            return []

        results: set[str] = set()
        self._collect_scheduler_enums(spec, hinted=False, results=results)

        if not results:
            return []

        return sorted(results)

    def _load_installed_schedulers(self, base_url: str) -> List[str]:
        """Attempt to retrieve the schedulers explicitly installed on the server."""

        endpoints = (
            "/api/v1/app/metadata",
            "/api/v1/metadata/schedulers",
            "/api/v1/metadata",
            "/api/v1/app/config",
            "/api/v1/app/configuration",
            "/api/v1/app/schedulers",
            "/api/v1/schedulers",
            "/api/v1/samplers",
        )

        seen: set[str] = set()
        ordered: List[str] = []
        last_error: Optional[requests.RequestException] = None

        for path in endpoints:
            url = f"{base_url}{path}"
            try:
                resp = requests.get(url, timeout=5)
            except requests.RequestException as exc:
                last_error = exc
                continue

            if resp.status_code and resp.status_code >= 400:
                continue

            payload = self._safe_json(resp)
            candidates = self._extract_installed_scheduler_names(payload)
            for name in candidates:
                if name not in seen:
                    seen.add(name)
                    ordered.append(name)

            if ordered:
                return ordered

        if last_error is not None:
            raise last_error

        return []

    def _extract_installed_scheduler_names(self, payload: Any) -> List[str]:
        """Extract scheduler names from various InvokeAI metadata payloads."""

        if payload is None:
            return []

        candidates: List[str] = []

        def contains_hint(node: Any) -> bool:
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(key, str):
                        lowered = key.lower()
                        if "scheduler" in lowered or "sampler" in lowered:
                            return True
                    if contains_hint(value):
                        return True
            elif isinstance(node, list):
                for item in node:
                    if contains_hint(item):
                        return True
            return False

        def add_name(value: Any) -> None:
            if isinstance(value, str):
                name = value.strip()
                if name:
                    candidates.append(name)

        def handle_collection(collection: Iterable[Any]) -> None:
            values: List[str] = []
            for item in collection:
                if isinstance(item, str):
                    text = item.strip()
                    if text:
                        values.append(text)
                elif isinstance(item, dict):
                    for key in ("name", "id", "scheduler", "value", "label"):
                        add_name(item.get(key))
                else:
                    continue
            if values:
                candidates.extend(values)

        if isinstance(payload, list):
            handle_collection(payload)
        elif isinstance(payload, dict):
            matched = False
            for key in (
                "schedulers",
                "scheduler_names",
                "scheduler_list",
                "available_schedulers",
                "installed_schedulers",
                "samplers",
                "data",
                "items",
                "result",
            ):
                if key in payload:
                    matched = True
                    names = self._extract_installed_scheduler_names(payload[key])
                    candidates.extend(names)
            if not candidates:
                details = payload.get("metadata")
                if isinstance(details, dict):
                    candidates.extend(self._extract_installed_scheduler_names(details))
                    matched = True
            if not candidates and not matched:
                for key, value in payload.items():
                    if not isinstance(value, (dict, list)):
                        continue
                    key_hint = isinstance(key, str) and (
                        "scheduler" in key.lower() or "sampler" in key.lower()
                    )
                    if key_hint or contains_hint(value):
                        candidates.extend(self._extract_installed_scheduler_names(value))

        filtered: List[str] = []
        seen: set[str] = set()
        for name in candidates:
            normalized = name.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                filtered.append(normalized)

        return filtered

    def _collect_scheduler_enums(self, node: Any, hinted: bool, results: set[str]) -> None:
        if isinstance(node, dict):
            local_hint = hinted
            for key in ("name", "title", "description"):
                value = node.get(key)
                if isinstance(value, str) and "scheduler" in value.lower():
                    local_hint = True
                    break

            enum_values = node.get("enum")
            if local_hint and isinstance(enum_values, list):
                for entry in enum_values:
                    if isinstance(entry, str):
                        text = entry.strip()
                        if text:
                            results.add(text)

            for key, value in node.items():
                next_hint = local_hint or (isinstance(key, str) and "scheduler" in key.lower())
                self._collect_scheduler_enums(value, next_hint, results)
        elif isinstance(node, list):
            for item in node:
                self._collect_scheduler_enums(item, hinted, results)

    def _build_base_urls(self, scheme: str, allow_https_fallback: bool) -> List[str]:
        normalized = (scheme or "http").lower().rstrip(":/")
        if normalized not in {"http", "https"}:
            raise ValueError("scheme must be 'http' or 'https'")

        base_urls = [f"{normalized}://{self.ip}:{self.port}"]
        if allow_https_fallback:
            alternate = "https" if normalized == "http" else "http"
            alt_url = f"{alternate}://{self.ip}:{self.port}"
            if alt_url not in base_urls:
                base_urls.append(alt_url)
        return base_urls

    def _promote_base_url(self, base_url: str) -> None:
        try:
            self._base_urls.remove(base_url)
        except ValueError:
            pass
        self._base_urls.insert(0, base_url)
        self.base_url = base_url

    def _parse_models_payload(self, payload: Any) -> Optional[List[InvokeAIModel]]:
        """Extract model metadata from an InvokeAI response payload."""

        items: Optional[List[Dict[str, Any]]] = None
        if isinstance(payload, list):
            items = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            for key in (
                "models",
                "items",
                "data",
                "results",
                "records",
                "entries",
            ):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    items = [item for item in candidate if isinstance(item, dict)]
                    break
                if isinstance(candidate, dict):
                    nested_items = []
                    for value in candidate.values():
                        if isinstance(value, list):
                            nested_items.extend(
                                [entry for entry in value if isinstance(entry, dict)]
                            )
                    if nested_items:
                        items = nested_items
                        break
            if items is None:
                for fallback_key in ("result", "available_models", "models_list"):
                    candidate = payload.get(fallback_key)
                    if isinstance(candidate, list):
                        items = [item for item in candidate if isinstance(item, dict)]
                        break

        if items is None:
            return None

        models: List[InvokeAIModel] = []
        skip_tokens = (
            "lora",
            "embedding",
            "textual",
            "vae",
            "control",
            "adapter",
            "clip",
            "ip_adapter",
            "t2i",
        )

        for item in items:
            model_type = item.get("type") or item.get("model_type")
            normalized_type = str(model_type).lower() if model_type is not None else ""
            if normalized_type and any(token in normalized_type for token in skip_tokens):
                continue
            name = item.get("name") or item.get("id") or item.get("key")
            if not name:
                name = item.get("model_name")
            if not name:
                continue
            base = (
                item.get("base")
                or item.get("base_model")
                or item.get("model_base")
                or ""
            ).lower()
            models.append(InvokeAIModel(name=name, base=base, key=item.get("key"), raw=item))

        return models

    def submit_image_generation(
        self,
        model: InvokeAIModel,
        prompt: str,
        negative_prompt: str = "",
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        scheduler: str = DEFAULT_SCHEDULER,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Submit a generation request and return queue identifiers."""

        payload, graph_info, seed_value = self._build_enqueue_payload(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            scheduler=scheduler,
            seed=seed,
        )

        enqueue_url = f"{self.base_url}/api/v1/queue/{QUEUE_ID}/enqueue_batch"
        resp = requests.post(enqueue_url, json=payload, timeout=15)
        resp.raise_for_status()

        data = self._safe_json(resp)
        item_id = self._extract_queue_item_id(data)
        if item_id is None:
            location_header = getattr(resp, "headers", {}).get("Location") if hasattr(resp, "headers") else None
            item_id = self._extract_queue_item_id_from_location(location_header)
            graph_id = graph_info["graph"].get("id") if isinstance(graph_info.get("graph"), dict) else None
            if graph_id and item_id == graph_id:
                item_id = None

        batch_info = data.get("batch") if isinstance(data, dict) else None
        batch_id = None
        if isinstance(batch_info, dict):
            batch_id = batch_info.get("batch_id") or batch_info.get("id")

        return {
            "queue_item_id": item_id,
            "batch_id": batch_id,
            "seed": seed_value,
            "graph": graph_info,
            "response": data,
        }

    def generate_image(
        self,
        model: InvokeAIModel,
        prompt: str,
        negative_prompt: str = "",
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        steps: int = DEFAULT_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        scheduler: str = DEFAULT_SCHEDULER,
        seed: Optional[int] = None,
        timeout: float = 180.0,
    ) -> Dict[str, Any]:
        """Generate an image and return metadata about the result."""

        payload, graph_info, seed_value = self._build_enqueue_payload(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            scheduler=scheduler,
            seed=seed,
        )

        enqueue_url = f"{self.base_url}/api/v1/queue/{QUEUE_ID}/enqueue_batch"
        enqueue_started = time.time()
        resp = requests.post(enqueue_url, json=payload, timeout=15)
        resp.raise_for_status()
        data = self._safe_json(resp)

        item_id = self._extract_queue_item_id(data)
        if item_id is None:
            location_header = getattr(resp, "headers", {}).get("Location") if hasattr(resp, "headers") else None
            item_id = self._extract_queue_item_id_from_location(location_header)
            graph_id = graph_info["graph"].get("id") if isinstance(graph_info.get("graph"), dict) else None
            if graph_id and item_id == graph_id:
                item_id = None

        batch_info = data.get("batch") if isinstance(data, dict) else None
        batch_id = None
        if isinstance(batch_info, dict):
            batch_id = batch_info.get("batch_id") or batch_info.get("id")

        session: Optional[Dict[str, Any]] = None
        batch_session_info: Optional[Tuple[Optional[str], Dict[str, Any]]] = None
        if item_id is not None:
            session = self._poll_queue(item_id=item_id, timeout=timeout)
        else:
            session = self._extract_session_from_enqueue_response(data)
            if session is None:
                batch_session_info = self._poll_queue_by_batch(
                    batch_id=batch_id,
                    timeout=timeout,
                    enqueue_started=enqueue_started,
                )
                if batch_session_info is not None:
                    item_id, session = batch_session_info
            if session is None:
                fallback_image = self._wait_for_board_image(
                    board_name="Auto",
                    enqueue_started=enqueue_started,
                    timeout=timeout,
                )
                if fallback_image is not None:
                    return self._build_result_from_board_image(
                        image_info=fallback_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed_value=seed_value,
                        width=width,
                        height=height,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        scheduler=scheduler,
                        model=model,
                    )
                raise InvokeAIClientError("InvokeAI did not return a queue item id")
            if item_id is None and batch_session_info is not None:
                item_id = batch_session_info[0]
            if item_id is None:
                item_id = self._extract_queue_item_id(session, allow_generic_id=False)

        output_node = graph_info["output"]
        result = self._extract_image_result(session, output_node)
        image_info = result.get("image") or result.get("images")
        image_meta: Optional[Dict[str, Any]] = None
        if isinstance(image_info, list) and image_info:
            image_meta = image_info[0]
        elif isinstance(image_info, dict):
            image_meta = image_info
        if not image_meta:
            raise InvokeAIClientError("InvokeAI response did not include image metadata")
        image_name = image_meta.get("image_name")
        if not image_name:
            raise InvokeAIClientError("InvokeAI response missing image name")

        return self._store_image_result(
            image_name=image_name,
            image_meta=image_meta,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed_value=seed_value,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            scheduler=scheduler,
            model=model,
            item_id=item_id,
            session_id=session.get("id") if isinstance(session, dict) else None,
        )

    def list_boards(self) -> List[Dict[str, Any]]:
        """Return the boards available on the InvokeAI server."""

        boards_url = f"{self.base_url}/api/v1/boards/"
        try:
            resp = requests.get(boards_url, params={"all": "true"}, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise InvokeAIClientError(f"Failed to retrieve boards: {exc}") from exc

        payload = self._safe_json(resp)
        if isinstance(payload, dict):
            items = payload.get("items")
            candidates = items if isinstance(items, list) else []
        elif isinstance(payload, list):
            candidates = payload
        else:
            candidates = []

        boards: List[Dict[str, Any]] = []
        uncategorized_seen = False
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            board_id = entry.get("board_id") or entry.get("id")
            name = entry.get("board_name") or entry.get("name")
            if isinstance(board_id, str) and board_id.strip():
                normalized_id = board_id.strip()
            else:
                normalized_id = None

            normalized_name = name.strip() if isinstance(name, str) else ""
            is_uncategorized = False
            if normalized_id:
                is_uncategorized = normalized_id.lower() == UNCATEGORIZED_BOARD_ID.lower()
            if not is_uncategorized and normalized_name:
                is_uncategorized = normalized_name.lower() == UNCATEGORIZED_BOARD_NAME.lower()

            if is_uncategorized:
                normalized_id = UNCATEGORIZED_BOARD_ID
                normalized_name = UNCATEGORIZED_BOARD_NAME
                uncategorized_seen = True

            if not normalized_id:
                # Skip unknown entries, but keep tracking uncategorized separately.
                continue

            display_name = normalized_name if normalized_name else normalized_id
            info: Dict[str, Any] = {
                "id": normalized_id,
                "name": display_name,
            }
            count = entry.get("image_count") or entry.get("count")
            if isinstance(count, (int, float)):
                info["count"] = int(count)
            if is_uncategorized:
                info["is_uncategorized"] = True
            boards.append(info)

        if not uncategorized_seen:
            boards.append(
                {
                    "id": UNCATEGORIZED_BOARD_ID,
                    "name": UNCATEGORIZED_BOARD_NAME,
                    "is_uncategorized": True,
                }
            )

        boards.sort(key=lambda value: value.get("name", "").lower())
        return boards

    def list_board_images(
        self,
        board_id: str,
        *,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Return the newest images for a board."""

        params: Dict[str, Any] = {
            "limit": max(1, int(limit)),
            "offset": max(0, int(offset)),
            "order_dir": "DESC",
            "starred_first": "false",
        }

        if board_id == UNCATEGORIZED_BOARD_ID:
            params["board_id"] = UNCATEGORIZED_BOARD_ID
            params["include_unassigned"] = "true"
        else:
            if not board_id:
                return []
            params["board_id"] = board_id

        images_url = f"{self.base_url}/api/v1/images/"
        try:
            resp = requests.get(images_url, params=params, timeout=15)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise InvokeAIClientError(f"Failed to retrieve board images: {exc}") from exc

        payload = self._safe_json(resp)
        if isinstance(payload, dict):
            items = payload.get("items")
            entries = items if isinstance(items, list) else []
        elif isinstance(payload, list):
            entries = payload
        else:
            entries = []

        return [entry for entry in entries if isinstance(entry, dict)]

    def retrieve_board_image(
        self,
        *,
        image_info: Dict[str, Any],
        board_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Download an image listed on a board for preview."""

        if not isinstance(image_info, dict):
            raise InvokeAIClientError("Board image entry must be a dictionary")

        image_name = image_info.get("image_name")
        if not isinstance(image_name, str) or not image_name.strip():
            raise InvokeAIClientError("Board image metadata missing image name")

        raw_metadata = image_info.get("metadata") if isinstance(image_info.get("metadata"), dict) else None
        metadata_payload = raw_metadata or self._fetch_image_metadata(image_name) or {}

        prompt = self._coerce_str(
            metadata_payload.get("positive_prompt"),
            fallback=self._coerce_str(metadata_payload.get("prompt"), default=""),
            default="",
        )
        if not prompt:
            prompt = self._coerce_str(image_info.get("prompt"), default="")

        negative_prompt = self._coerce_str(
            metadata_payload.get("negative_prompt"),
            fallback=self._coerce_str(image_info.get("negative_prompt"), default=""),
            default="",
        )

        width = self._coerce_int(
            self._extract_value(metadata_payload, image_info, "width"),
            default=DEFAULT_WIDTH,
        )
        height = self._coerce_int(
            self._extract_value(metadata_payload, image_info, "height"),
            default=DEFAULT_HEIGHT,
        )
        steps = self._coerce_int(
            self._extract_value(metadata_payload, image_info, "steps"),
            default=DEFAULT_STEPS,
        )
        cfg_scale = self._coerce_float(
            self._extract_value(metadata_payload, image_info, "cfg_scale"),
            default=DEFAULT_CFG_SCALE,
        )
        scheduler = self._coerce_str(
            self._extract_value(metadata_payload, image_info, "scheduler")
            or self._extract_value(metadata_payload, image_info, "sampler"),
            default=DEFAULT_SCHEDULER,
        )
        seed_value = self._coerce_int(
            self._extract_value(metadata_payload, image_info, "seed"),
            default=random.randint(0, 2**31 - 1),
        )

        model_info = self._extract_model(metadata_payload, image_info)
        model = InvokeAIModel(
            name=model_info.get("name", "(unknown)"),
            base=model_info.get("base", "unknown"),
            key=model_info.get("key"),
            raw=model_info.get("raw", {}),
        )

        image_meta = dict(image_info)
        if board_name:
            image_meta.setdefault("board", board_name)
        if metadata_payload:
            image_meta.setdefault("metadata", metadata_payload)

        return self._store_image_result(
            image_name=image_name,
            image_meta=image_meta,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed_value=seed_value,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            scheduler=scheduler,
            model=model,
            item_id=None,
            session_id=None,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_enqueue_payload(
        self,
        *,
        model: InvokeAIModel,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        scheduler: str,
        seed: Optional[int],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
        if not prompt.strip():
            raise InvokeAIClientError("Prompt must not be empty")

        seed_value = seed if seed is not None else random.randint(0, 2**31 - 1)
        graph_info = self._build_graph(
            model=model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            scheduler=scheduler,
            seed=seed_value,
        )

        batch_payload: Dict[str, Any] = {
            "graph": graph_info["graph"],
            "origin": ORIGIN,
            "destination": DESTINATION,
            "runs": 1,
        }
        if graph_info.get("data") is not None:
            batch_payload["data"] = graph_info["data"]

        body = {
            "prepend": False,
            "batch": batch_payload,
        }

        return body, graph_info, seed_value

    def _coerce_str(
        self,
        value: Any,
        *,
        fallback: Optional[str] = None,
        default: str = "",
    ) -> str:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        if isinstance(fallback, str):
            text = fallback.strip()
            if text:
                return text
        return default

    def _coerce_int(self, value: Any, *, default: int) -> int:
        try:
            if isinstance(value, bool):
                raise TypeError
            return int(value)
        except (TypeError, ValueError):
            return default

    def _coerce_float(self, value: Any, *, default: float) -> float:
        try:
            if isinstance(value, bool):
                raise TypeError
            return float(value)
        except (TypeError, ValueError):
            return default

    def _extract_value(
        self,
        metadata_payload: Optional[Dict[str, Any]],
        image_info: Optional[Dict[str, Any]],
        key: str,
    ) -> Any:
        if isinstance(metadata_payload, dict) and key in metadata_payload:
            return metadata_payload.get(key)
        if isinstance(image_info, dict) and key in image_info:
            return image_info.get(key)
        if isinstance(image_info, dict):
            nested = image_info.get("metadata")
            if isinstance(nested, dict) and key in nested:
                return nested.get(key)
        return None

    def _extract_model(
        self,
        metadata_payload: Optional[Dict[str, Any]],
        image_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        candidate: Any = None
        if isinstance(metadata_payload, dict):
            candidate = metadata_payload.get("model")
            if candidate is None:
                candidate = metadata_payload.get("model_settings")
        if candidate is None and isinstance(image_info, dict):
            candidate = image_info.get("model")

        raw: Dict[str, Any] = {}
        name = "(unknown)"
        base = "unknown"
        key = None

        if isinstance(candidate, dict):
            raw = dict(candidate)
            name = self._coerce_str(
                candidate.get("name")
                or candidate.get("model_name")
                or candidate.get("model"),
                default="(unknown)",
            )
            base = self._coerce_str(
                candidate.get("base")
                or candidate.get("base_model")
                or candidate.get("type"),
                default="unknown",
            )
            key_value = candidate.get("key")
            key = key_value if isinstance(key_value, str) else None
        elif isinstance(candidate, str):
            name = self._coerce_str(candidate, default="(unknown)")
            meta_base = None
            if isinstance(metadata_payload, dict):
                meta_base = metadata_payload.get("base_model")
            base = self._coerce_str(meta_base, default="unknown")
        else:
            if isinstance(metadata_payload, dict):
                name = self._coerce_str(
                    metadata_payload.get("model_name")
                    or metadata_payload.get("model"),
                    default="(unknown)",
                )
                base = self._coerce_str(
                    metadata_payload.get("base_model")
                    or metadata_payload.get("model_base"),
                    default="unknown",
                )

        return {
            "name": name,
            "base": (base or "unknown").lower(),
            "key": key,
            "raw": raw,
        }

    def _poll_queue(self, item_id: Any, timeout: float) -> Dict[str, Any]:
        item_id_str = str(item_id).strip()
        if not item_id_str:
            raise InvokeAIClientError("Invalid queue item id")

        status_url = f"{self.base_url}/api/v1/queue/{QUEUE_ID}/i/{item_id_str}"
        start = time.time()
        last_status = None
        while True:
            if time.time() - start > timeout:
                raise InvokeAIClientError(
                    f"Timed out after {int(timeout)}s waiting for queue item {item_id}"
                )
            resp = requests.get(status_url, timeout=10)
            resp.raise_for_status()
            payload = self._safe_json(resp)
            if not isinstance(payload, dict):
                raise InvokeAIClientError(
                    "InvokeAI queue status response was not valid JSON"
                )
            status = payload.get("status")
            last_status = status
            if status == "completed":
                session = payload.get("session")
                if not isinstance(session, dict):
                    raise InvokeAIClientError("Queue item completed without session data")
                return session
            if status in {"failed", "canceled"}:
                message = payload.get("error_message") or payload.get("error_type") or status
                raise InvokeAIClientError(f"Queue item {item_id} {status}: {message}")
            time.sleep(2)

    def _poll_queue_by_batch(
        self,
        *,
        batch_id: Optional[str],
        timeout: float,
        enqueue_started: float,
    ) -> Optional[Tuple[Optional[str], Dict[str, Any]]]:
        if not batch_id:
            return None

        status_url = f"{self.base_url}/api/v1/queue/{QUEUE_ID}/b/{batch_id}/status"
        deadline = enqueue_started + timeout
        last_error: Optional[InvokeAIClientError] = None

        while time.time() < deadline:
            try:
                resp = requests.get(status_url, timeout=10)
                if resp.status_code == 404:
                    break
                resp.raise_for_status()
            except requests.RequestException:
                time.sleep(2)
                continue

            payload = self._safe_json(resp)
            if not isinstance(payload, dict):
                time.sleep(2)
                continue

            failed_count = payload.get("failed")
            if isinstance(failed_count, int) and failed_count > 0:
                message = payload.get("error_message") or "batch failed"
                last_error = InvokeAIClientError(
                    f"Queue batch {batch_id} failed: {message}"
                )
                break

            total = payload.get("total")
            completed = payload.get("completed")
            if (
                isinstance(total, int)
                and total > 0
                and isinstance(completed, int)
                and completed >= total
            ):
                for item in self._list_queue_items_for_batch(batch_id):
                    if not isinstance(item, dict):
                        continue
                    session = item.get("session") if isinstance(item.get("session"), dict) else None
                    status = item.get("status")
                    if status != "completed" or session is None:
                        continue
                    queue_item = item.get("item_id")
                    queue_item_str = str(queue_item).strip() if queue_item is not None else None
                    return queue_item_str, session
                break

            time.sleep(2)

        if last_error is not None:
            raise last_error
        return None

    def _list_queue_items_for_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        list_url = f"{self.base_url}/api/v1/queue/{QUEUE_ID}/list_all"
        attempts = [{"destination": DESTINATION}, {}]
        for params in attempts:
            try:
                resp = requests.get(list_url, params=params or None, timeout=15)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
            except requests.RequestException:
                continue

            payload = self._safe_json(resp)
            items: List[Dict[str, Any]] = []
            if isinstance(payload, list):
                items = [entry for entry in payload if isinstance(entry, dict)]
            elif isinstance(payload, dict):
                maybe_items = payload.get("items")
                if isinstance(maybe_items, list):
                    items = [entry for entry in maybe_items if isinstance(entry, dict)]

            matches: List[Dict[str, Any]] = []
            for entry in items:
                entry_batch = entry.get("batch_id")
                if entry_batch is None:
                    continue
                if str(entry_batch) != str(batch_id):
                    continue
                matches.append(entry)

            if matches:
                return matches

        return []

    def _wait_for_board_image(
        self,
        *,
        board_name: str,
        enqueue_started: float,
        timeout: float,
    ) -> Optional[Dict[str, Any]]:
        board_id = self._fetch_board_id(board_name)

        fetchers: List[Callable[[], Optional[Dict[str, Any]]]] = []
        if board_id:
            def fetch_board() -> Optional[Dict[str, Any]]:
                return self._fetch_latest_board_image(board_id)

            fetchers.append(fetch_board)

        fetchers.append(self._fetch_latest_global_image)

        deadline = enqueue_started + timeout
        threshold = enqueue_started - 5
        while time.time() < deadline:
            for fetcher in fetchers:
                try:
                    image_info = fetcher()
                except Exception:
                    image_info = None
                if not isinstance(image_info, dict):
                    continue

                created_value = (
                    image_info.get("created_at")
                    or image_info.get("updated_at")
                    or image_info.get("timestamp")
                )
                created_at = self._parse_iso_timestamp(created_value)
                if created_at is None or created_at >= threshold:
                    return image_info

            time.sleep(2)

        return None

    def _fetch_board_id(self, board_name: str) -> Optional[str]:
        name_norm = board_name.strip().lower()
        if not name_norm:
            return None

        boards_url = f"{self.base_url}/api/v1/boards/"
        try:
            resp = requests.get(boards_url, params={"all": "true"}, timeout=15)
            resp.raise_for_status()
        except requests.RequestException:
            return None

        payload = self._safe_json(resp)
        candidates: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            candidates = [entry for entry in payload if isinstance(entry, dict)]
        elif isinstance(payload, dict):
            items = payload.get("items")
            if isinstance(items, list):
                candidates = [entry for entry in items if isinstance(entry, dict)]

        for entry in candidates:
            label = entry.get("board_name") or entry.get("name")
            if isinstance(label, str) and label.strip().lower() == name_norm:
                board_id = entry.get("board_id") or entry.get("id")
                if isinstance(board_id, str) and board_id.strip():
                    return board_id
        return None

    def _fetch_latest_board_image(self, board_id: str) -> Optional[Dict[str, Any]]:
        return self._fetch_latest_image({"board_id": board_id})

    def _fetch_latest_global_image(self) -> Optional[Dict[str, Any]]:
        return self._fetch_latest_image(None)

    def _fetch_latest_image(
        self, extra_params: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        images_url = f"{self.base_url}/api/v1/images/"
        params: Dict[str, Any] = {
            "limit": 1,
            "offset": 0,
            "order_dir": "DESC",
            "starred_first": "false",
        }
        if extra_params:
            params.update(extra_params)
        try:
            resp = requests.get(images_url, params=params, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except requests.RequestException:
            return None

        payload = self._safe_json(resp)
        items: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            potential = payload.get("items")
            if isinstance(potential, list):
                items = [entry for entry in potential if isinstance(entry, dict)]
        elif isinstance(payload, list):
            items = [entry for entry in payload if isinstance(entry, dict)]

        return items[0] if items else None

    def _parse_iso_timestamp(self, value: Any) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime.datetime):
            dt = value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                dt = datetime.datetime.fromisoformat(text)
            except ValueError:
                return None
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        return None

    def _fetch_image_metadata(self, image_name: str) -> Optional[Dict[str, Any]]:
        metadata_url = f"{self.base_url}/api/v1/images/i/{image_name}/metadata"
        try:
            resp = requests.get(metadata_url, timeout=15)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except requests.RequestException:
            return None

        payload = self._safe_json(resp)
        return payload if isinstance(payload, dict) else None

    def _normalize_metadata_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._normalize_metadata_value(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._normalize_metadata_value(item) for item in value]
        if isinstance(value, datetime.datetime):
            dt = value if value.tzinfo else value.replace(tzinfo=datetime.timezone.utc)
            return dt.isoformat()
        return value

    def _store_image_result(
        self,
        *,
        image_name: str,
        image_meta: Dict[str, Any],
        prompt: str,
        negative_prompt: str,
        seed_value: int,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        scheduler: str,
        model: InvokeAIModel,
        item_id: Optional[str],
        session_id: Optional[str],
    ) -> Dict[str, Any]:
        image_url = f"{self.base_url}/api/v1/images/i/{image_name}/full"
        image_resp = requests.get(image_url, timeout=30)
        image_resp.raise_for_status()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = image_name.replace("/", "_")
        out_path = self.images_dir / f"{timestamp}_{safe_name}"
        with out_path.open("wb") as f:
            f.write(image_resp.content)

        meta_path = out_path.with_suffix(".json")
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed_value,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "scheduler": scheduler,
            "model": {
                "name": model.name,
                "base": model.base,
                "key": model.key,
            },
            "server": {
                "ip": self.ip,
                "port": self.port,
                "nickname": self.nickname,
            },
            "queue_item": item_id,
            "session_id": session_id,
            "image": self._normalize_metadata_value(image_meta),
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return {
            "path": out_path,
            "metadata_path": meta_path,
            "image_name": image_name,
            "metadata": metadata,
        }

    def _build_result_from_board_image(
        self,
        *,
        image_info: Dict[str, Any],
        prompt: str,
        negative_prompt: str,
        seed_value: int,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        scheduler: str,
        model: InvokeAIModel,
    ) -> Dict[str, Any]:
        image_name = image_info.get("image_name") if isinstance(image_info, dict) else None
        if not isinstance(image_name, str) or not image_name.strip():
            raise InvokeAIClientError("Latest board image did not include an image name")

        image_meta = dict(image_info)
        metadata_payload = self._fetch_image_metadata(image_name)
        if metadata_payload is not None:
            image_meta.setdefault("metadata", metadata_payload)

        raw_session_id = image_info.get("session_id")
        session_id = raw_session_id if isinstance(raw_session_id, str) else None

        return self._store_image_result(
            image_name=image_name,
            image_meta=image_meta,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed_value=seed_value,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            scheduler=scheduler,
            model=model,
            item_id=None,
            session_id=session_id,
        )

    def _extract_image_result(self, session: Dict[str, Any], output_node: str) -> Dict[str, Any]:
        results = session.get("results")
        if not isinstance(results, dict):
            raise InvokeAIClientError("Session results missing or invalid")
        mapping = session.get("source_prepared_mapping", {})
        prepared_ids = []
        if isinstance(mapping, dict):
            prepared_ids = mapping.get(output_node) or []
        candidate_ids = list(prepared_ids) if isinstance(prepared_ids, list) else []
        if not candidate_ids:
            candidate_ids.append(output_node)
        for candidate in candidate_ids:
            result = results.get(candidate)
            if isinstance(result, dict):
                return result
        # Fallback: return first image-like result
        for result in results.values():
            if isinstance(result, dict) and (
                "image" in result or "images" in result or result.get("type") == "image"
            ):
                return result
        raise InvokeAIClientError("No image result found in session data")

    def _build_graph(
        self,
        model: InvokeAIModel,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        scheduler: str,
        seed: int,
    ) -> Dict[str, Any]:
        base = model.base.lower()
        if base in {"sd-1", "sd1", "sd-2", "sd2", "stable-diffusion-1", "stable-diffusion-2"}:
            return self._build_sd_graph(model.raw, prompt, negative_prompt, width, height, steps, cfg_scale, scheduler, seed)
        if base in {"sdxl", "sdxl-refiner", "stable-diffusion-xl"}:
            return self._build_sdxl_graph(model.raw, prompt, negative_prompt, width, height, steps, cfg_scale, scheduler, seed)
        raise InvokeAIClientError(f"Unsupported InvokeAI base model: {base or 'unknown'}")

    def _build_sd_graph(
        self,
        model_cfg: Dict[str, Any],
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        scheduler: str,
        seed: int,
    ) -> Dict[str, Any]:
        graph_id = "terminal_sd_graph"
        clip_skip_layers = model_cfg.get("clip_skip", 0)
        use_clip_skip = isinstance(clip_skip_layers, int) and clip_skip_layers > 0

        nodes: Dict[str, Dict[str, Any]] = {
            "model_loader": {
                "id": "model_loader",
                "type": "main_model_loader",
                "model": model_cfg,
            },
            "positive_conditioning": {
                "id": "positive_conditioning",
                "type": "compel",
                "prompt": prompt,
            },
            "negative_conditioning": {
                "id": "negative_conditioning",
                "type": "compel",
                "prompt": negative_prompt or "",
            },
            "noise": {
                "id": "noise",
                "type": "noise",
                "seed": seed,
                "use_cpu": False,
                "width": width,
                "height": height,
            },
            "denoise": {
                "id": "denoise",
                "type": "denoise_latents",
                "cfg_scale": cfg_scale,
                "cfg_rescale_multiplier": 0.0,
                "scheduler": scheduler,
                "steps": steps,
                "denoising_start": 0.0,
                "denoising_end": 1.0,
            },
            "latents_to_image": {
                "id": "latents_to_image",
                "type": "l2i",
                "fp32": False,
            },
        }

        edges: List[Dict[str, Dict[str, str]]] = []
        if use_clip_skip:
            nodes["clip_skip"] = {
                "id": "clip_skip",
                "type": "clip_skip",
                "skipped_layers": clip_skip_layers,
            }
            edges.append(
                {
                    "source": {"node_id": "model_loader", "field": "clip"},
                    "destination": {"node_id": "clip_skip", "field": "clip"},
                }
            )
            clip_source = {"node_id": "clip_skip", "field": "clip"}
        else:
            clip_source = {"node_id": "model_loader", "field": "clip"}

        edges.extend(
            [
                {
                    "source": clip_source,
                    "destination": {"node_id": "positive_conditioning", "field": "clip"},
                },
                {
                    "source": clip_source,
                    "destination": {"node_id": "negative_conditioning", "field": "clip"},
                },
                {
                    "source": {"node_id": "model_loader", "field": "unet"},
                    "destination": {"node_id": "denoise", "field": "unet"},
                },
                {
                    "source": {"node_id": "model_loader", "field": "vae"},
                    "destination": {"node_id": "latents_to_image", "field": "vae"},
                },
                {
                    "source": {"node_id": "positive_conditioning", "field": "conditioning"},
                    "destination": {"node_id": "denoise", "field": "positive_conditioning"},
                },
                {
                    "source": {"node_id": "negative_conditioning", "field": "conditioning"},
                    "destination": {"node_id": "denoise", "field": "negative_conditioning"},
                },
                {
                    "source": {"node_id": "noise", "field": "noise"},
                    "destination": {"node_id": "denoise", "field": "noise"},
                },
                {
                    "source": {"node_id": "denoise", "field": "latents"},
                    "destination": {"node_id": "latents_to_image", "field": "latents"},
                },
            ]
        )

        graph = {"id": graph_id, "nodes": nodes, "edges": edges}
        return {"graph": graph, "data": None, "output": "latents_to_image"}

    def _extract_queue_item_id(
        self, payload: Any, *, allow_generic_id: bool = True
    ) -> Optional[str]:
        """Extract a queue item identifier from an InvokeAI enqueue response."""

        if payload is None:
            return None

        # Prioritize explicit queue identifiers before considering generic ids.
        queue_keys: List[str] = [
            "queue_item_id",
            "queue_item_ids",
            "item_id",
            "item_ids",
        ]
        if allow_generic_id:
            queue_keys.extend(["id", "ids"])

        queue_keys = [self._canonicalize_queue_key(key) for key in queue_keys]

        to_visit: List[Any] = [payload]
        while to_visit:
            current = to_visit.pop(0)
            if isinstance(current, dict):
                is_session_like = self._looks_like_session_payload(current)
                keys_by_priority: List[Tuple[str, Any]] = []
                for actual_key, value in current.items():
                    canonical_key = self._canonicalize_queue_key(actual_key)
                    if canonical_key not in queue_keys:
                        continue
                    keys_by_priority.append((canonical_key, value))

                # Preserve preference ordering defined in queue_keys.
                for canonical_key in queue_keys:
                    for matched_key, value in keys_by_priority:
                        if matched_key != canonical_key:
                            continue
                        if canonical_key in {"id", "ids"}:
                            if is_session_like:
                                continue
                            # Avoid mistaking graph or node identifiers for queue ids.
                            if {"nodes", "edges"} & current.keys():
                                continue
                            if "type" in current and "status" not in current:
                                continue
                        normalized = self._normalize_queue_item_id(
                            value, allow_generic_id=allow_generic_id
                        )
                        if normalized is not None:
                            return normalized
                        break

                for value in current.values():
                    if isinstance(value, (dict, list)):
                        to_visit.append(value)
            elif isinstance(current, list):
                to_visit.extend(item for item in current if isinstance(item, (dict, list)))

        return None

    def _canonicalize_queue_key(self, key: str) -> str:
        return "".join(ch for ch in key.lower() if ch.isalnum())

    def _normalize_queue_item_id(
        self, value: Any, *, allow_generic_id: bool = True
    ) -> Optional[str]:
        if isinstance(value, list):
            for entry in value:
                normalized = self._normalize_queue_item_id(
                    entry, allow_generic_id=allow_generic_id
                )
                if normalized is not None:
                    return normalized
            return None
        if isinstance(value, dict):
            return self._extract_queue_item_id(value, allow_generic_id=allow_generic_id)
        if isinstance(value, (int, str)):
            text = str(value).strip()
            if text:
                return text
        return None

    def _extract_queue_item_id_from_location(self, location: Optional[str]) -> Optional[str]:
        if not isinstance(location, str):
            return None

        candidate = location.strip()
        if not candidate:
            return None

        candidate = candidate.split("?", 1)[0].rstrip("/")
        if not candidate:
            return None

        if "/i/" in candidate:
            candidate = candidate.rsplit("/i/", 1)[-1]
        else:
            candidate = candidate.rsplit("/", 1)[-1]

        candidate = candidate.strip().strip("/")
        if candidate:
            return candidate
        return None

    def _extract_session_from_enqueue_response(
        self, payload: Any
    ) -> Optional[Dict[str, Any]]:
        if payload is None:
            return None

        to_visit: List[Any] = [payload]
        while to_visit:
            current = to_visit.pop(0)
            if isinstance(current, dict):
                if self._looks_like_session_payload(current):
                    return current

                session = current.get("session")
                if isinstance(session, dict):
                    return session

                sessions = current.get("sessions")
                if isinstance(sessions, list):
                    for entry in sessions:
                        if isinstance(entry, dict):
                            return entry

                for value in current.values():
                    if isinstance(value, (dict, list)):
                        to_visit.append(value)
            elif isinstance(current, list):
                to_visit.extend(item for item in current if isinstance(item, (dict, list)))

        return None

    def _looks_like_session_payload(self, candidate: Dict[str, Any]) -> bool:
        if not isinstance(candidate, dict):
            return False

        results = candidate.get("results")
        if not isinstance(results, dict):
            return False

        for key in ("id", "session_id", "queue_item_id", "item_id"):
            if key in candidate:
                return True

        return False

    def _build_sdxl_graph(
        self,
        model_cfg: Dict[str, Any],
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        scheduler: str,
        seed: int,
    ) -> Dict[str, Any]:
        graph_id = "terminal_sdxl_graph"
        nodes: Dict[str, Dict[str, Any]] = {
            "model_loader": {
                "id": "model_loader",
                "type": "sdxl_model_loader",
                "model": model_cfg,
            },
            "positive_conditioning": {
                "id": "positive_conditioning",
                "type": "sdxl_compel_prompt",
                "prompt": prompt,
                "style": prompt,
                "original_width": width,
                "original_height": height,
                "target_width": width,
                "target_height": height,
                "crop_top": 0,
                "crop_left": 0,
            },
            "negative_conditioning": {
                "id": "negative_conditioning",
                "type": "sdxl_compel_prompt",
                "prompt": negative_prompt or "",
                "style": negative_prompt or "",
                "original_width": width,
                "original_height": height,
                "target_width": width,
                "target_height": height,
                "crop_top": 0,
                "crop_left": 0,
            },
            "noise": {
                "id": "noise",
                "type": "noise",
                "seed": seed,
                "use_cpu": False,
                "width": width,
                "height": height,
            },
            "denoise": {
                "id": "denoise",
                "type": "denoise_latents",
                "cfg_scale": cfg_scale,
                "cfg_rescale_multiplier": 0.0,
                "scheduler": scheduler,
                "steps": steps,
                "denoising_start": 0.0,
                "denoising_end": 1.0,
            },
            "latents_to_image": {
                "id": "latents_to_image",
                "type": "l2i",
                "fp32": False,
            },
        }

        edges = [
            {
                "source": {"node_id": "model_loader", "field": "clip"},
                "destination": {"node_id": "positive_conditioning", "field": "clip"},
            },
            {
                "source": {"node_id": "model_loader", "field": "clip"},
                "destination": {"node_id": "negative_conditioning", "field": "clip"},
            },
            {
                "source": {"node_id": "model_loader", "field": "clip2"},
                "destination": {"node_id": "positive_conditioning", "field": "clip2"},
            },
            {
                "source": {"node_id": "model_loader", "field": "clip2"},
                "destination": {"node_id": "negative_conditioning", "field": "clip2"},
            },
            {
                "source": {"node_id": "model_loader", "field": "unet"},
                "destination": {"node_id": "denoise", "field": "unet"},
            },
            {
                "source": {"node_id": "model_loader", "field": "vae"},
                "destination": {"node_id": "latents_to_image", "field": "vae"},
            },
            {
                "source": {"node_id": "positive_conditioning", "field": "conditioning"},
                "destination": {"node_id": "denoise", "field": "positive_conditioning"},
            },
            {
                "source": {"node_id": "negative_conditioning", "field": "conditioning"},
                "destination": {"node_id": "denoise", "field": "negative_conditioning"},
            },
            {
                "source": {"node_id": "noise", "field": "noise"},
                "destination": {"node_id": "denoise", "field": "noise"},
            },
            {
                "source": {"node_id": "denoise", "field": "latents"},
                "destination": {"node_id": "latents_to_image", "field": "latents"},
            },
        ]

        graph = {"id": graph_id, "nodes": nodes, "edges": edges}
        return {"graph": graph, "data": None, "output": "latents_to_image"}


__all__ = ["InvokeAIClient", "InvokeAIClientError", "InvokeAIModel"]
