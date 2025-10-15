"""Client helpers for interacting with Automatic1111's Stable Diffusion API."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from image_cache import (
    ensure_images_dir,
    normalize_metadata_value,
    resolve_cached_paths,
    write_metadata_file,
)


class Automatic1111ClientError(RuntimeError):
    """Raised when an Automatic1111 operation fails."""


@dataclass
class Automatic1111Model:
    """Representation of an Automatic1111 checkpoint."""

    name: str
    title: str
    model_hash: Optional[str]
    raw: Dict[str, Any]


class Automatic1111Client:
    """Small helper for the Automatic1111 REST API."""

    def __init__(
        self,
        ip: str,
        port: int,
        nickname: Optional[str] = None,
        data_dir: Optional[Path] = None,
        scheme: str = "http",
    ) -> None:
        self.ip = ip
        self.port = port
        self.nickname = nickname or ip
        self.base_url = f"{scheme}://{ip}:{port}"
        base_data = data_dir if data_dir is not None else Path(__file__).resolve().parent.parent / "data"
        self.images_dir = ensure_images_dir(Path(base_data))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check_health(self) -> bool:
        """Return True if the Automatic1111 API responds to basic queries."""

        try:
            resp = requests.get(f"{self.base_url}/sdapi/v1/sd-models", timeout=5)
            resp.raise_for_status()
        except requests.RequestException:
            return False

        payload = self._safe_json(resp)
        return isinstance(payload, list)

    def list_models(self) -> List[Automatic1111Model]:
        """Return the list of available checkpoints on the server."""

        resp = requests.get(f"{self.base_url}/sdapi/v1/sd-models", timeout=10)
        resp.raise_for_status()
        payload = self._safe_json(resp)
        if not isinstance(payload, list):
            raise Automatic1111ClientError("Automatic1111 returned an invalid models list")

        models: List[Automatic1111Model] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            name = self._coerce_str(
                entry.get("model_name")
                or entry.get("model")
                or entry.get("name")
                or entry.get("title")
            )
            title = self._coerce_str(entry.get("title") or name)
            if not name:
                continue
            model_hash = self._coerce_str(entry.get("hash") or entry.get("sha256"))
            models.append(
                Automatic1111Model(
                    name=name,
                    title=title or name,
                    model_hash=model_hash,
                    raw=dict(entry),
                )
            )

        if not models:
            raise Automatic1111ClientError("Automatic1111 did not provide any models")

        models.sort(key=lambda m: m.title.lower())
        return models

    def list_samplers(self) -> List[str]:
        """Return the known sampler names supported by the server."""

        try:
            resp = requests.get(f"{self.base_url}/sdapi/v1/samplers", timeout=10)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise Automatic1111ClientError(self._summarize_http_error(exc)) from exc

        payload = self._safe_json(resp)
        options: List[str] = []
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, str):
                    text = entry.strip()
                    if text:
                        options.append(text)
                elif isinstance(entry, dict):
                    text = self._coerce_str(entry.get("name") or entry.get("title"))
                    if text:
                        options.append(text)

        # Deduplicate while preserving order
        deduped: List[str] = []
        seen: set[str] = set()
        for option in options:
            lowered = option.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(option)
        return deduped

    def set_active_model(self, model: Automatic1111Model) -> None:
        """Select the checkpoint that should be used for future generations."""

        checkpoint = model.title or model.name
        payload = {"sd_model_checkpoint": checkpoint}

        try:
            resp = requests.post(
                f"{self.base_url}/sdapi/v1/options",
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise Automatic1111ClientError(self._summarize_http_error(exc)) from exc

    def txt2img(
        self,
        *,
        model: Automatic1111Model,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        sampler: Optional[str],
        seed: Optional[int],
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Submit a text-to-image generation request and cache the first result."""

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "n_iter": 1,
            "batch_size": 1,
        }
        if sampler:
            payload["sampler_name"] = sampler
        if seed is not None:
            payload["seed"] = int(seed)

        try:
            resp = requests.post(
                f"{self.base_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise Automatic1111ClientError(self._summarize_http_error(exc)) from exc

        data = self._safe_json(resp)
        if not isinstance(data, dict):
            raise Automatic1111ClientError("Automatic1111 returned an invalid generation payload")

        images = data.get("images")
        if not isinstance(images, list) or not images:
            raise Automatic1111ClientError("Automatic1111 response did not include any images")

        raw_image = images[0]
        if not isinstance(raw_image, str) or not raw_image.strip():
            raise Automatic1111ClientError("Automatic1111 returned an empty image payload")

        image_bytes = self._decode_image(raw_image)
        image_name = self._build_image_name(image_bytes, seed)
        out_path, meta_path = resolve_cached_paths(self.images_dir, image_name)
        image_already_cached = out_path.exists()

        if not image_already_cached:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("wb") as handle:
                handle.write(image_bytes)

        parameters = data.get("parameters") if isinstance(data.get("parameters"), dict) else {}
        parsed_info, raw_info = self._parse_info(data.get("info"))

        seed_used = self._extract_seed(parameters, parsed_info, seed)
        sampler_used = self._extract_sampler(parameters, parsed_info, sampler)
        width_used = self._extract_dimension(parameters, parsed_info, "width", width)
        height_used = self._extract_dimension(parameters, parsed_info, "height", height)
        steps_used = self._extract_numeric(parameters, parsed_info, "steps", steps)
        cfg_used = self._extract_numeric(parameters, parsed_info, "cfg_scale", cfg_scale)

        metadata: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed_used,
            "width": width_used,
            "height": height_used,
            "steps": steps_used,
            "cfg_scale": cfg_used,
            "sampler": sampler_used,
            "model": {
                "name": model.name,
                "title": model.title,
                "hash": model.model_hash,
            },
            "server": {
                "ip": self.ip,
                "port": self.port,
                "nickname": self.nickname,
            },
            "parameters": normalize_metadata_value(parameters),
            "info": normalize_metadata_value(parsed_info),
        }
        if raw_info:
            metadata["raw_info"] = raw_info

        write_metadata_file(meta_path, metadata)

        return {
            "path": out_path,
            "metadata_path": meta_path,
            "metadata": metadata,
            "cached": True,
            "from_cache": image_already_cached,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _safe_json(self, response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return None

    def _summarize_http_error(self, exc: requests.RequestException) -> str:
        message = str(exc)
        response = getattr(exc, "response", None)
        if response is None:
            return message
        try:
            payload = response.json()
        except ValueError:
            payload = None
        detail = None
        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("error")
            if isinstance(detail, (list, tuple)) and detail:
                detail = detail[0]
        if not detail and isinstance(response.content, (bytes, bytearray)):
            text = response.content.decode(errors="ignore").strip()
            if text:
                detail = text[:200]
        return f"{message} ({detail})" if detail else message

    def _decode_image(self, payload: str) -> bytes:
        text = payload.strip()
        if "," in text and text.split(",", 1)[0].startswith("data:"):
            text = text.split(",", 1)[1]
        try:
            return base64.b64decode(text, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise Automatic1111ClientError("Failed to decode image payload from Automatic1111") from exc

    def _build_image_name(self, image_bytes: bytes, seed: Optional[int]) -> str:
        digest = hashlib.sha256(image_bytes).hexdigest()[:16]
        suffix = f"_{seed}" if seed is not None else ""
        timestamp = int(time.time())
        return f"automatic1111_{digest}{suffix}_{timestamp}.png"

    def _parse_info(self, info_payload: Any) -> tuple[Dict[str, Any], Optional[str]]:
        if isinstance(info_payload, dict):
            return dict(info_payload), None
        if isinstance(info_payload, str):
            stripped = info_payload.strip()
            if not stripped:
                return {}, None
            try:
                parsed = json.loads(stripped)
            except ValueError:
                return {}, stripped
            if isinstance(parsed, dict):
                return parsed, stripped
            return {}, stripped
        return {}, None

    def _extract_seed(
        self,
        parameters: Dict[str, Any],
        info: Dict[str, Any],
        seed: Optional[int],
    ) -> Optional[int]:
        for key in ("Seed", "seed"):
            value = parameters.get(key)
            coerced = self._coerce_int(value)
            if coerced is not None:
                return coerced
        info_seed = info.get("seed")
        coerced_info = self._coerce_int(info_seed)
        if coerced_info is not None:
            return coerced_info
        seeds = info.get("all_seeds") or info.get("All seeds") or info.get("allSeeds")
        if isinstance(seeds, list) and seeds:
            coerced_list = self._coerce_int(seeds[0])
            if coerced_list is not None:
                return coerced_list
        return seed

    def _extract_sampler(
        self,
        parameters: Dict[str, Any],
        info: Dict[str, Any],
        sampler: Optional[str],
    ) -> Optional[str]:
        for key in ("Sampler", "sampler", "sampler_name"):
            value = self._coerce_str(parameters.get(key))
            if value:
                return value
        info_sampler = self._coerce_str(info.get("sampler_name") or info.get("sampler"))
        if info_sampler:
            return info_sampler
        samplers = info.get("all_samplers") or info.get("All samplers")
        if isinstance(samplers, list) and samplers:
            text = self._coerce_str(samplers[0])
            if text:
                return text
        return sampler

    def _extract_dimension(
        self,
        parameters: Dict[str, Any],
        info: Dict[str, Any],
        key: str,
        fallback: int,
    ) -> int:
        for candidate in (key, key.title()):
            value = parameters.get(candidate)
            coerced = self._coerce_int(value)
            if coerced is not None:
                return coerced
        info_value = info.get(key)
        coerced_info = self._coerce_int(info_value)
        if coerced_info is not None:
            return coerced_info
        return int(fallback)

    def _extract_numeric(
        self,
        parameters: Dict[str, Any],
        info: Dict[str, Any],
        key: str,
        fallback: float,
    ) -> float:
        for candidate in (key, key.title(), key.replace("_", " ").title()):
            value = parameters.get(candidate)
            coerced = self._coerce_float(value)
            if coerced is not None:
                return coerced
        info_value = info.get(key)
        coerced_info = self._coerce_float(info_value)
        if coerced_info is not None:
            return coerced_info
        return float(fallback)

    def _coerce_int(self, value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return None
        return None

    def _coerce_float(self, value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    def _coerce_str(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        return ""
