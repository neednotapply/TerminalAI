"""Helper utilities for interacting with InvokeAI servers."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_SCHEDULER = "dpmpp_2m"
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

        return sorted(DEFAULT_SCHEDULER_OPTIONS)

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

        enqueue_url = f"{self.base_url}/api/v1/queue/{QUEUE_ID}/enqueue_batch"
        resp = requests.post(enqueue_url, json=body, timeout=15)
        resp.raise_for_status()
        data = self._safe_json(resp)

        item_id = self._extract_queue_item_id(data)
        if item_id is None:
            location_header = getattr(resp, "headers", {}).get("Location") if hasattr(resp, "headers") else None
            item_id = self._extract_queue_item_id_from_location(location_header)

        session: Optional[Dict[str, Any]] = None
        if item_id is not None:
            session = self._poll_queue(item_id=item_id, timeout=timeout)
        else:
            session = self._extract_session_from_enqueue_response(data)
            if session is None:
                raise InvokeAIClientError("InvokeAI did not return a queue item id")
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
            "session_id": session.get("id"),
            "image": image_meta,
        }
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return {
            "path": out_path,
            "metadata_path": meta_path,
            "image_name": image_name,
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
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

        queue_keys = {
            "item_id",
            "item_ids",
            "queue_item_id",
            "queue_item_ids",
        }
        if allow_generic_id:
            queue_keys.update({"id", "ids"})

        to_visit: List[Any] = [payload]
        while to_visit:
            current = to_visit.pop(0)
            if isinstance(current, dict):
                is_session_like = self._looks_like_session_payload(current)
                for key, value in current.items():
                    if key in queue_keys:
                        if is_session_like and key in {"id", "ids"}:
                            continue
                        normalized = self._normalize_queue_item_id(
                            value, allow_generic_id=allow_generic_id
                        )
                        if normalized is not None:
                            return normalized
                    if isinstance(value, (dict, list)):
                        to_visit.append(value)
            elif isinstance(current, list):
                to_visit.extend(item for item in current if isinstance(item, (dict, list)))

        return None

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
