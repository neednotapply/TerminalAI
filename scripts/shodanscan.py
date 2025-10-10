import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import shodan
import socket
import time
import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning

try:
    from invoke_client import STATIC_MODEL_ENDPOINTS
except ImportError:  # pragma: no cover - fallback for packaging edge cases
    def _build_static_model_endpoints() -> List[Tuple[str, Optional[Dict[str, Any]]]]:
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

    STATIC_MODEL_ENDPOINTS = _build_static_model_endpoints()

urllib3.disable_warnings(InsecureRequestWarning)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LEGACY_CSV_PATH = DATA_DIR / "endpoints.csv"
OLLAMA_CSV_PATH = DATA_DIR / "ollama.endpoints.csv"
INVOKE_CSV_PATH = DATA_DIR / "invoke.endpoints.csv"
CSV_PATHS = {
    "ollama": OLLAMA_CSV_PATH,
    "invokeai": INVOKE_CSV_PATH,
}
CONFIG_PATH = DATA_DIR / "config.json"
SHODAN_QUERIES = [
    {
        "query": 'http.html:"Ollama is running" port:11434',
        "api_type": "ollama",
        "default_port": 11434,
    },
    {
        "query": 'title:"Invoke - Community Edition"',
        "api_type": "invokeai",
        "default_port": 9090,
    },
]

BASE_COLUMNS = [
    "id",
    "ip",
    "port",
    "scan_date",
    "verified",
    "verification_date",
    "is_active",
    "inactive_reason",
    "last_check_date",
    "api_type",
    "hostnames",
    "org",
    "isp",
    "city",
    "region",
    "country",
    "latitude",
    "longitude",
    "ping",
]

COLUMN_ORDER = {
    "ollama": BASE_COLUMNS,
    "invokeai": BASE_COLUMNS + ["available_models"],
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_name(city, country, org):
    org = org or ""
    org = (org[:17] + "...") if len(org) > 20 else org
    location = ", ".join([p for p in [city, country] if p])
    if org and location:
        return f"{location} ({org})"
    return location or org


def normalise_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def ensure_columns(df, api_type):
    df = df.copy()
    if "api_type" in df.columns:
        df["api_type"] = df["api_type"].astype(str).str.lower()
        df = df[df["api_type"] == api_type]
    else:
        df["api_type"] = api_type

    columns = COLUMN_ORDER.get(api_type, BASE_COLUMNS)
    for col in columns:
        if col not in df.columns:
            if col == "verified":
                df[col] = 0
            elif col == "is_active":
                df[col] = True
            else:
                df[col] = ""
    if "available_models" in columns and "available_models" not in df.columns:
        df["available_models"] = ""

    df["verified"] = pd.to_numeric(df["verified"], errors="coerce").fillna(0).astype(int)
    df["is_active"] = df["is_active"].apply(normalise_bool)
    df["port"] = pd.to_numeric(df["port"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ip", "port"])
    df["port"] = df["port"].astype(int)
    df["ping"] = pd.to_numeric(df["ping"], errors="coerce")
    return df


def load_dataframe(api_type):
    candidates = []
    primary = CSV_PATHS.get(api_type)
    if primary:
        candidates.append(primary)
    if api_type == "ollama":
        candidates.append(LEGACY_CSV_PATH)

    for path in candidates:
        if not path or not path.exists():
            continue
        try:
            df = pd.read_csv(path, keep_default_na=False)
        except Exception as exc:
            logging.warning("Failed to read %s: %s", path, exc)
            continue
        return ensure_columns(df, api_type)

    df = pd.DataFrame(columns=COLUMN_ORDER.get(api_type, BASE_COLUMNS))
    df["api_type"] = api_type
    return ensure_columns(df, api_type)


def save_dataframe(api_type, df):
    out = df.copy()
    out["api_type"] = api_type
    columns = COLUMN_ORDER.get(api_type, BASE_COLUMNS)
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    ordered = columns + [c for c in out.columns if c not in columns]
    out = out[ordered]
    path = CSV_PATHS[api_type]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_api_key():
    logging.info("Loading Shodan API key")
    try:
        with CONFIG_PATH.open() as f:
            key = json.load(f).get("SHODAN_API_KEY")
            if key:
                logging.info("Loaded API key from config file")
    except (OSError, json.JSONDecodeError):
        logging.info("Failed to load API key from config file")
        key = None
    if not key:
        key = os.getenv("SHODAN_API_KEY")
        if key:
            logging.info("Loaded API key from environment variable")
    if not key:
        logging.info("Shodan API key not found")
    return key


def ping_time(ip, port):
    """Return the TCP connect latency in ms or None if unreachable."""
    try:
        start = time.time()
        with socket.create_connection((ip, int(port)), timeout=1):
            end = time.time()
        return (end - start) * 1000
    except Exception:
        return None


def extract_primary_hostname(value):
    if not value:
        return None
    try:
        if pd.isna(value):  # type: ignore[attr-defined]
            return None
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        for item in value:
            primary = extract_primary_hostname(item)
            if primary:
                return primary
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(";") if part.strip()]
        return parts[0] if parts else None
    return str(value).strip() or None


def check_ollama_api(ip, port, hostname=None):
    """Check that the Ollama API responds on the given host and port."""
    url = f"http://{ip}:{port}/api/tags"
    try:
        headers = {"Host": hostname} if hostname else None
        r = requests.get(url, timeout=2, headers=headers)
        if r.status_code == 200:
            try:
                data = r.json()
                models = [
                    m.get("name") or m.get("model") or m.get("id")
                    for m in data.get("models", [])
                ]
                models = [m for m in models if m]
            except Exception:
                models = []
            return True, "", models
        return False, f"http {r.status_code}", []
    except requests.RequestException as e:
        return False, str(e), []


def _parse_invoke_models(payload):
    if isinstance(payload, list):
        models = []
        for item in payload:
            if isinstance(item, dict):
                model_type = item.get("type") or item.get("model_type")
                normalized_type = str(model_type).lower() if model_type is not None else ""
                if normalized_type:
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
                    if any(token in normalized_type for token in skip_tokens):
                        continue
                    if not any(
                        keyword in normalized_type
                        for keyword in ("main", "onnx", "checkpoint", "model", "base", "core")
                    ):
                        continue
                name = (
                    item.get("name")
                    or item.get("id")
                    or item.get("key")
                    or item.get("model_name")
                )
                if not name:
                    continue
                base = (
                    item.get("base")
                    or item.get("base_model")
                    or item.get("model_base")
                )
                models.append(f"{name} ({base})" if base else name)
            elif isinstance(item, str):
                name = item.strip()
                if name:
                    models.append(name)
        return models
    if isinstance(payload, dict):
        for key in ("models", "items", "data", "results"):
            nested = payload.get(key)
            models = _parse_invoke_models(nested)
            if models:
                return models
            if isinstance(nested, dict):
                collected = []
                for value in nested.values():
                    sub_models = _parse_invoke_models(value)
                    if sub_models:
                        collected.extend(sub_models)
                if collected:
                    return collected
    return []


def _discover_invoke_model_endpoints(
    base: str, headers: Optional[Dict[str, str]], verify: bool
) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
    discovered: List[Tuple[str, Optional[Dict[str, Any]]]] = []
    seen: set[Tuple[str, Tuple[Tuple[str, Any], ...]]] = set()
    for suffix in ("/openapi.json", "/docs/openapi.json"):
        url = f"{base}{suffix}"
        try:
            resp = requests.get(url, timeout=5, headers=headers, verify=verify)
            resp.raise_for_status()
        except requests.RequestException:
            continue

        try:
            spec = resp.json()
        except ValueError:
            continue

        if not isinstance(spec, dict):
            continue

        paths = spec.get("paths")
        if not isinstance(paths, dict):
            continue

        for raw_path, operations in paths.items():
            if not isinstance(raw_path, str):
                continue
            lowered = raw_path.lower()
            if "model" not in lowered:
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
                if param.get("name") != "model_type":
                    continue
                schema = param.get("schema") if isinstance(param.get("schema"), dict) else {}
                values = schema.get("enum") if isinstance(schema.get("enum"), list) else []
                options = [v for v in values if isinstance(v, str)]
                if not options:
                    options = ["main"]
                for value in options:
                    if value.lower() in {"main", "checkpoint", "primary"}:
                        param_candidates.append({"model_type": value})

            for param in param_candidates:
                variants = {raw_path}
                if raw_path.endswith("/"):
                    variants.add(raw_path.rstrip("/"))
                else:
                    variants.add(f"{raw_path}/")
                for variant in variants:
                    param_items = tuple(sorted(param.items())) if param else tuple()
                    key = (variant, param_items)
                    if key in seen:
                        continue
                    seen.add(key)
                    discovered.append((variant, param))

    return discovered


def check_invoke_api(ip, port, hostname=None):
    """Check that the InvokeAI API responds and gather available models."""

    last_error = ""
    targets = []

    if hostname:
        targets.append((hostname, {"Host": hostname}))

    # Always fall back to the raw IP in case DNS for the hostname no longer
    # resolves to the scanned address.
    targets.append((ip, None))

    for scheme in ("http", "https"):
        verify_tls = scheme != "https"

        for target, headers in targets:
            base = f"{scheme}://{target}:{port}"
            version_url = f"{base}/api/v1/app/version"

            try:
                r = requests.get(
                    version_url,
                    timeout=5,
                    headers=headers,
                    verify=verify_tls,
                )
                r.raise_for_status()
            except requests.RequestException as e:
                last_error = str(e)
                continue

            model_candidates: List[Tuple[str, Optional[Dict[str, Any]]]] = list(
                STATIC_MODEL_ENDPOINTS
            )
            model_candidates.extend(
                _discover_invoke_model_endpoints(base, headers, verify_tls)
            )

            seen_candidates: set[Tuple[str, Tuple[Tuple[str, Any], ...]]] = set()

            for path, params in model_candidates:
                normalized_path = path if path.startswith("/") else f"/{path}"
                param_items = tuple(sorted(params.items())) if params else tuple()
                key = (normalized_path, param_items)
                if key in seen_candidates:
                    continue
                seen_candidates.add(key)

                url = f"{base}{normalized_path}"
                try:
                    resp = requests.get(
                        url,
                        params=params,
                        timeout=5,
                        headers=headers,
                        verify=verify_tls,
                    )
                    resp.raise_for_status()
                except requests.HTTPError as exc:
                    status = (
                        exc.response.status_code
                        if exc.response is not None
                        else None
                    )
                    if status == 404:
                        continue
                    last_error = str(exc)
                    break
                except requests.RequestException as exc:
                    last_error = str(exc)
                    break

                try:
                    payload = resp.json()
                except ValueError:
                    payload = None

                models = _parse_invoke_models(payload)
                if models:
                    return True, "", models
            else:
                return True, "models fetch failed: no compatible endpoint", []

    return False, last_error or "version check failed", []


def update_existing(api, df, batch_size, api_type):
    if df.empty:
        return df

    # Batch the IPs by port to minimise API requests.
    grouped = {}
    for _, row in df.iterrows():
        port = int(row["port"])
        grouped.setdefault(port, set()).add(row["ip"])

    details = {}
    errors = {}
    for port, ips in grouped.items():
        ip_list = list(ips)
        for i in range(0, len(ip_list), batch_size):
            chunk = ip_list[i : i + batch_size]
            ip_query = " OR ".join(f"ip:{ip}" for ip in chunk)
            query = f"port:{port} ({ip_query})"
            try:
                results = (
                    api.search(query, limit=batch_size).get("matches", [])
                )
                for r in results:
                    key = (r.get("ip_str"), r.get("port"))
                    details[key] = r
            except shodan.APIError as e:
                for ip in chunk:
                    errors[(ip, port)] = str(e)

    now = utc_now()
    for idx, row in df.iterrows():
        ip = row["ip"]
        port = int(row["port"])
        key = (ip, port)
        r = details.get(key)
        if r:
            location = r.get("location", {})
            df.at[idx, "is_active"] = True
            df.at[idx, "inactive_reason"] = ""
            df.at[idx, "last_check_date"] = now
            df.at[idx, "scan_date"] = r.get("timestamp", row.get("scan_date", now))
            df.at[idx, "verified"] = 1
            df.at[idx, "verification_date"] = now
            df.at[idx, "hostnames"] = ";".join(r.get("hostnames", []))
            org = r.get("org", "")
            df.at[idx, "org"] = org
            df.at[idx, "isp"] = r.get("isp", "")
            city = location.get("city", "")
            df.at[idx, "city"] = city
            df.at[idx, "region"] = location.get("region_code") or location.get("region_name", "")
            country = location.get("country_name", "")
            df.at[idx, "country"] = country
            df.at[idx, "latitude"] = location.get("latitude", "")
            df.at[idx, "longitude"] = location.get("longitude", "")
            df.at[idx, "id"] = build_name(city, country, org)
        else:
            df.at[idx, "is_active"] = False
            df.at[idx, "inactive_reason"] = errors.get(key, "port closed")
            df.at[idx, "last_check_date"] = now
        latency = ping_time(ip, port)
        df.at[idx, "ping"] = latency if latency is not None else pd.NA
        if latency is None:
            df.at[idx, "is_active"] = False
            df.at[idx, "inactive_reason"] = "ping timeout"
            if "available_models" in df.columns:
                df.at[idx, "available_models"] = ""
            continue

        hostname = extract_primary_hostname(
            df.at[idx, "hostnames"] if "hostnames" in df.columns else None
        )
        if api_type == "invokeai":
            api_ok, reason, models = check_invoke_api(ip, port, hostname=hostname)
        else:
            api_ok, reason, models = check_ollama_api(ip, port, hostname=hostname)
        df.at[idx, "is_active"] = api_ok
        df.at[idx, "inactive_reason"] = "" if api_ok else reason or "api error"
        if "available_models" in df.columns:
            df.at[idx, "available_models"] = ";".join(models)
    return df


def find_new(api, df, query_info, limit):
    api_type = query_info["api_type"]
    query = query_info["query"]
    default_port = query_info.get("default_port")
    existing = set(zip(df["ip"], df["port"]))
    new_rows = []

    logging.info(f"Executing Shodan query: {query} (limit {limit})")
    try:
        results = api.search(query, limit=limit * 5).get("matches", [])
    except shodan.APIError as e:
        logging.info(f"Shodan query failed: {e}")
        return df

    for r in results:
        ip = r.get("ip_str")
        port = r.get("port") or default_port
        if not ip or port is None:
            continue
        try:
            port = int(port)
        except (TypeError, ValueError):
            continue
        if (ip, port) in existing:
            continue
        location = r.get("location", {})
        scan_date = r.get("timestamp", utc_now())
        org = r.get("org", "")
        city = location.get("city", "")
        country = location.get("country_name", "")
        new_rows.append(
            {
                "id": build_name(city, country, org),
                "ip": ip,
                "port": port,
                "scan_date": scan_date,
                "verified": 0,
                "verification_date": "",
                "is_active": True,
                "inactive_reason": "",
                "last_check_date": scan_date,
                "api_type": api_type,
                "hostnames": ";".join(r.get("hostnames", [])),
                "org": org,
                "isp": r.get("isp", ""),
                "city": city,
                "region": location.get("region_code")
                or location.get("region_name", ""),
                "country": country,
                "latitude": location.get("latitude", ""),
                "longitude": location.get("longitude", ""),
                "ping": pd.NA,
            }
        )
        existing.add((ip, port))
        if len(new_rows) >= limit:
            break

    if not new_rows:
        return df

    new_df = ensure_columns(pd.DataFrame(new_rows), api_type)
    for idx, row in new_df.iterrows():
        latency = ping_time(row["ip"], row["port"])
        if latency is None:
            new_df.at[idx, "ping"] = pd.NA
            new_df.at[idx, "is_active"] = False
            new_df.at[idx, "inactive_reason"] = "ping timeout"
            if "available_models" in new_df.columns:
                new_df.at[idx, "available_models"] = ""
        else:
            hostname = extract_primary_hostname(row.get("hostnames"))
            if api_type == "invokeai":
                api_ok, reason, models = check_invoke_api(
                    row["ip"], row["port"], hostname=hostname
                )
            else:
                api_ok, reason, models = check_ollama_api(
                    row["ip"], row["port"], hostname=hostname
                )
            new_df.at[idx, "ping"] = latency
            new_df.at[idx, "is_active"] = api_ok
            new_df.at[idx, "inactive_reason"] = "" if api_ok else reason or "api error"
            if "available_models" in new_df.columns:
                new_df.at[idx, "available_models"] = ";".join(models)

    new_df["ping"] = pd.to_numeric(new_df["ping"], errors="coerce")
    new_df.sort_values(by="ping", inplace=True, na_position="last")

    if df.empty:
        return new_df

    combined = pd.concat([df, new_df], ignore_index=True)
    combined["ping"] = pd.to_numeric(combined["ping"], errors="coerce")
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Discover and verify Ollama and InvokeAI endpoints via Shodan"
    )
    parser.add_argument(
        "--debug",
        "-d",
        "--verbose",
        dest="debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum new endpoints to append per Shodan query",
    )
    parser.add_argument(
        "--existing-limit",
        type=int,
        default=25,
        help="Maximum existing endpoints to verify per run",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(message)s",
    )

    key = load_api_key()
    if not key:
        raise RuntimeError(
            "SHODAN_API_KEY not provided in config.json or environment variable"
        )
    api = shodan.Shodan(key)
    queries_by_type = {}
    for info in SHODAN_QUERIES:
        queries_by_type.setdefault(info["api_type"], []).append(info)

    for api_type, query_infos in queries_by_type.items():
        logging.info("Processing %s endpoints", api_type)
        df = load_dataframe(api_type)
        if not df.empty and "last_check_date" in df.columns:
            logging.info("Updating existing %s endpoints", api_type)
            oldest_idx = (
                df.sort_values(by="last_check_date")
                .head(args.existing_limit)
                .index
            )
            df.loc[oldest_idx] = update_existing(
                api, df.loc[oldest_idx].copy(), args.existing_limit, api_type
            )
            logging.info("Finished updating %s endpoints", api_type)
        else:
            logging.info("No existing %s endpoints to update", api_type)

        for info in query_infos:
            df = find_new(api, df, info, args.limit)

        columns = COLUMN_ORDER.get(api_type, BASE_COLUMNS)
        df = df[columns]
        df["ping"] = pd.to_numeric(df["ping"], errors="coerce")
        save_dataframe(api_type, df)


if __name__ == "__main__":
    main()
