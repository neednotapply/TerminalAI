"""Helper utilities for interacting with InvokeAI servers."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

DEFAULT_SCHEDULER = "dpmpp_2m"
DEFAULT_CFG_SCALE = 7.5
DEFAULT_STEPS = 30
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
QUEUE_ID = "default"
ORIGIN = "terminalai"
DESTINATION = "terminalai"


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

    def __init__(self, ip: str, port: int, nickname: Optional[str] = None, data_dir: Optional[Path] = None) -> None:
        self.ip = ip
        self.port = port
        self.nickname = nickname or ip
        self.base_url = f"http://{ip}:{port}"
        base_data = data_dir if data_dir is not None else Path(__file__).resolve().parent.parent / "data"
        self.images_dir = Path(base_data) / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check_health(self) -> Dict[str, Any]:
        """Ensure the InvokeAI invocation API is reachable."""

        version_url = f"{self.base_url}/api/v1/app/version"
        queue_url = f"{self.base_url}/api/v1/queue/{QUEUE_ID}/size"

        try:
            version_resp = requests.get(version_url, timeout=5)
            version_resp.raise_for_status()
        except requests.RequestException as exc:
            raise InvokeAIClientError("Failed to query InvokeAI version endpoint") from exc

        try:
            queue_resp = requests.get(queue_url, timeout=5)
            queue_resp.raise_for_status()
        except requests.RequestException as exc:
            raise InvokeAIClientError("Failed to query InvokeAI queue endpoint") from exc

        version_payload = self._safe_json(version_resp)
        queue_payload = self._safe_json(queue_resp)

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

        candidates = [
            ("/api/v2/models", {"model_type": "main"}),
            ("/api/v1/models", {"model_type": "main"}),
            ("/api/v1/models", {"type": "main"}),
            ("/api/v1/models/main", None),
        ]
        last_http_error: Optional[requests.HTTPError] = None

        for path, params in candidates:
            url = f"{self.base_url}{path}"
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status == 404:
                    last_http_error = exc
                    continue
                raise

            models = self._parse_models_payload(resp.json())
            if models is None:
                continue
            models.sort(key=lambda m: m.name.lower())
            return models

        if last_http_error is not None:
            raise InvokeAIClientError(
                "InvokeAI server did not expose a compatible models endpoint"
            ) from last_http_error

        raise InvokeAIClientError("InvokeAI server did not return a valid models list")

    def _safe_json(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return None

    def _parse_models_payload(self, payload: Any) -> Optional[List[InvokeAIModel]]:
        """Extract model metadata from an InvokeAI response payload."""

        items: Optional[List[Dict[str, Any]]] = None
        if isinstance(payload, list):
            items = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            for key in ("models", "items", "data"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    items = [item for item in candidate if isinstance(item, dict)]
                    break
            if items is None and "results" in payload and isinstance(payload["results"], list):
                items = [item for item in payload["results"] if isinstance(item, dict)]

        if items is None:
            return None

        models: List[InvokeAIModel] = []
        for item in items:
            model_type = item.get("type") or item.get("model_type")
            if model_type not in (None, "main", "Main"):
                continue
            name = item.get("name") or item.get("id") or item.get("key")
            if not name:
                continue
            base = (item.get("base") or item.get("base_model") or "").lower()
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

        body = {
            "prepend": False,
            "batch": {
                "graph": graph_info["graph"],
                "data": graph_info["batch"],
                "origin": ORIGIN,
                "destination": DESTINATION,
                "runs": 1,
            },
        }

        enqueue_url = f"{self.base_url}/api/v1/queue/{QUEUE_ID}/enqueue_batch"
        resp = requests.post(enqueue_url, json=body, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        item_ids = data.get("item_ids", [])
        if not item_ids:
            raise InvokeAIClientError("InvokeAI did not return a queue item id")
        item_id = item_ids[0]

        session = self._poll_queue(item_id=item_id, timeout=timeout)
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
    def _poll_queue(self, item_id: int, timeout: float) -> Dict[str, Any]:
        status_url = f"{self.base_url}/api/v1/queue/{QUEUE_ID}/i/{item_id}"
        start = time.time()
        last_status = None
        while True:
            if time.time() - start > timeout:
                raise InvokeAIClientError(
                    f"Timed out after {int(timeout)}s waiting for queue item {item_id}"
                )
            resp = requests.get(status_url, timeout=10)
            resp.raise_for_status()
            payload = resp.json()
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
        nodes: Dict[str, Dict[str, Any]] = {
            "positive_prompt": {"id": "positive_prompt", "type": "string"},
            "neg_prompt": {
                "id": "neg_prompt",
                "type": "compel",
                "prompt": negative_prompt or "",
            },
            "pos_collect": {"id": "pos_collect", "type": "collect"},
            "neg_collect": {"id": "neg_collect", "type": "collect"},
            "seed": {"id": "seed", "type": "integer"},
            "noise": {
                "id": "noise",
                "type": "noise",
                "use_cpu": False,
                "width": width,
                "height": height,
            },
            "model_loader": {
                "id": "model_loader",
                "type": "main_model_loader",
                "model": model_cfg,
            },
            "clip_skip": {
                "id": "clip_skip",
                "type": "clip_skip",
                "skipped_layers": model_cfg.get("clip_skip", 0),
            },
            "pos_cond": {"id": "pos_cond", "type": "compel"},
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
            "l2i": {"id": "l2i", "type": "l2i", "fp32": False},
        }

        edges = [
            {"source": {"node_id": "model_loader", "field": "clip"}, "destination": {"node_id": "clip_skip", "field": "clip"}},
            {"source": {"node_id": "clip_skip", "field": "clip"}, "destination": {"node_id": "pos_cond", "field": "clip"}},
            {"source": {"node_id": "clip_skip", "field": "clip"}, "destination": {"node_id": "neg_prompt", "field": "clip"}},
            {"source": {"node_id": "model_loader", "field": "unet"}, "destination": {"node_id": "denoise", "field": "unet"}},
            {"source": {"node_id": "model_loader", "field": "vae"}, "destination": {"node_id": "l2i", "field": "vae"}},
            {"source": {"node_id": "positive_prompt", "field": "value"}, "destination": {"node_id": "pos_cond", "field": "prompt"}},
            {"source": {"node_id": "pos_cond", "field": "conditioning"}, "destination": {"node_id": "pos_collect", "field": "item"}},
            {"source": {"node_id": "pos_collect", "field": "collection"}, "destination": {"node_id": "denoise", "field": "positive_conditioning"}},
            {"source": {"node_id": "neg_prompt", "field": "conditioning"}, "destination": {"node_id": "neg_collect", "field": "item"}},
            {"source": {"node_id": "neg_collect", "field": "collection"}, "destination": {"node_id": "denoise", "field": "negative_conditioning"}},
            {"source": {"node_id": "seed", "field": "value"}, "destination": {"node_id": "noise", "field": "seed"}},
            {"source": {"node_id": "noise", "field": "noise"}, "destination": {"node_id": "denoise", "field": "noise"}},
            {"source": {"node_id": "denoise", "field": "latents"}, "destination": {"node_id": "l2i", "field": "latents"}},
        ]

        graph = {"id": graph_id, "nodes": nodes, "edges": edges}
        batch = [
            [
                {"node_path": "positive_prompt", "field_name": "value", "items": [prompt]},
                {"node_path": "seed", "field_name": "value", "items": [seed]},
            ]
        ]
        return {"graph": graph, "batch": batch, "output": "l2i"}

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
            "positive_prompt": {"id": "positive_prompt", "type": "string"},
            "pos_cond": {"id": "pos_cond", "type": "sdxl_compel_prompt"},
            "neg_cond": {
                "id": "neg_cond",
                "type": "sdxl_compel_prompt",
                "prompt": negative_prompt or "",
                "style": negative_prompt or "",
            },
            "pos_collect": {"id": "pos_collect", "type": "collect"},
            "neg_collect": {"id": "neg_collect", "type": "collect"},
            "seed": {"id": "seed", "type": "integer"},
            "noise": {
                "id": "noise",
                "type": "noise",
                "use_cpu": False,
                "width": width,
                "height": height,
            },
            "model_loader": {
                "id": "model_loader",
                "type": "sdxl_model_loader",
                "model": model_cfg,
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
            "l2i": {"id": "l2i", "type": "l2i", "fp32": False},
        }

        edges = [
            {"source": {"node_id": "model_loader", "field": "clip"}, "destination": {"node_id": "pos_cond", "field": "clip"}},
            {"source": {"node_id": "model_loader", "field": "clip"}, "destination": {"node_id": "neg_cond", "field": "clip"}},
            {"source": {"node_id": "model_loader", "field": "clip2"}, "destination": {"node_id": "pos_cond", "field": "clip2"}},
            {"source": {"node_id": "model_loader", "field": "clip2"}, "destination": {"node_id": "neg_cond", "field": "clip2"}},
            {"source": {"node_id": "model_loader", "field": "unet"}, "destination": {"node_id": "denoise", "field": "unet"}},
            {"source": {"node_id": "model_loader", "field": "vae"}, "destination": {"node_id": "l2i", "field": "vae"}},
            {"source": {"node_id": "positive_prompt", "field": "value"}, "destination": {"node_id": "pos_cond", "field": "prompt"}},
            {"source": {"node_id": "positive_prompt", "field": "value"}, "destination": {"node_id": "pos_cond", "field": "style"}},
            {"source": {"node_id": "pos_cond", "field": "conditioning"}, "destination": {"node_id": "pos_collect", "field": "item"}},
            {"source": {"node_id": "pos_collect", "field": "collection"}, "destination": {"node_id": "denoise", "field": "positive_conditioning"}},
            {"source": {"node_id": "neg_cond", "field": "conditioning"}, "destination": {"node_id": "neg_collect", "field": "item"}},
            {"source": {"node_id": "neg_collect", "field": "collection"}, "destination": {"node_id": "denoise", "field": "negative_conditioning"}},
            {"source": {"node_id": "seed", "field": "value"}, "destination": {"node_id": "noise", "field": "seed"}},
            {"source": {"node_id": "noise", "field": "noise"}, "destination": {"node_id": "denoise", "field": "noise"}},
            {"source": {"node_id": "denoise", "field": "latents"}, "destination": {"node_id": "l2i", "field": "latents"}},
        ]

        graph = {"id": graph_id, "nodes": nodes, "edges": edges}
        batch = [
            [
                {"node_path": "positive_prompt", "field_name": "value", "items": [prompt]},
                {"node_path": "seed", "field_name": "value", "items": [seed]},
            ]
        ]
        return {"graph": graph, "batch": batch, "output": "l2i"}


__all__ = ["InvokeAIClient", "InvokeAIClientError", "InvokeAIModel"]
