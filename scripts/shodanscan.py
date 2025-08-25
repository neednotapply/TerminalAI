import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import shodan
import socket
import time
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "endpoints.csv"
CONFIG_PATH = DATA_DIR / "config.json"
SHODAN_QUERIES = [
    'http.html:"Ollama is running" port:11434'
]

COLUMNS = [
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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_name(city, country, org):
    org = org or ""
    org = (org[:17] + "...") if len(org) > 20 else org
    location = ", ".join([p for p in [city, country] if p])
    if org and location:
        return f"{location} ({org})"
    return location or org


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


def check_ollama_api(ip, port):
    """Check that the Ollama API responds on the given host and port."""
    url = f"http://{ip}:{port}/api/tags"
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            return True, ""
        return False, f"http {r.status_code}"
    except requests.RequestException as e:
        return False, str(e)


def update_existing(api, df, batch_size):
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
        if latency is None:
            df.at[idx, "ping"] = pd.NA
            df.at[idx, "is_active"] = False
            df.at[idx, "inactive_reason"] = "ping timeout"
        else:
            api_ok, reason = check_ollama_api(ip, port)
            df.at[idx, "ping"] = latency
            df.at[idx, "is_active"] = api_ok
            df.at[idx, "inactive_reason"] = "" if api_ok else reason or "api error"
    return df


def find_new(api, df, limit):
    existing = set(zip(df["ip"], df["port"]))
    new_rows = []
    for query in SHODAN_QUERIES:
        logging.info(f"Executing Shodan query: {query} (limit {limit})")
        try:
            results = api.search(query, limit=limit * 5).get("matches", [])
        except shodan.APIError as e:
            logging.info(f"Shodan query failed: {e}")
            continue
        count = 0
        for r in results:
            ip = r.get("ip_str")
            port = r.get("port")
            if (ip, str(port)) in existing:
                continue
            location = r.get("location", {})
            scan_date = r.get("timestamp", utc_now())
            org = r.get("org", "")
            city = location.get("city", "")
            country = location.get("country_name", "")
            new_rows.append({
                "id": build_name(city, country, org),
                "ip": ip,
                "port": port,
                "scan_date": scan_date,
                "verified": 0,
                "verification_date": "",
                "is_active": True,
                "inactive_reason": "",
                "last_check_date": scan_date,
                "api_type": "ollama",
                "hostnames": ";".join(r.get("hostnames", [])),
                "org": org,
                "isp": r.get("isp", ""),
                "city": city,
                "region": location.get("region_code") or location.get("region_name", ""),
                "country": country,
                "latitude": location.get("latitude", ""),
                "longitude": location.get("longitude", ""),
                "ping": pd.NA,
            })
            existing.add((ip, str(port)))
            count += 1
        logging.info(f"Processed {count} results for query: {query}")
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        for idx, row in new_df.iterrows():
            latency = ping_time(row["ip"], row["port"])
            if latency is None:
                new_df.at[idx, "ping"] = pd.NA
                new_df.at[idx, "is_active"] = False
                new_df.at[idx, "inactive_reason"] = "ping timeout"
            else:
                api_ok, reason = check_ollama_api(row["ip"], row["port"])
                new_df.at[idx, "ping"] = latency
                new_df.at[idx, "is_active"] = api_ok
                new_df.at[idx, "inactive_reason"] = "" if api_ok else reason or "api error"
        new_df["ping"] = pd.to_numeric(new_df["ping"], errors="coerce")
        new_df.sort_values(by="ping", inplace=True, na_position="last")
        new_df = new_df.head(limit)
        if df.empty:
            df = new_df
        else:
            df = pd.concat([df, new_df], ignore_index=True)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Keep endpoints.csv up to date using the Shodan API"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
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
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s:%(message)s",
    )

    key = load_api_key()
    if not key:
        raise RuntimeError(
            "SHODAN_API_KEY not provided in config.json or environment variable"
        )
    api = shodan.Shodan(key)
    try:
        df = pd.read_csv(CSV_PATH, keep_default_na=False)
    except FileNotFoundError:
        df = pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["ping"] = pd.to_numeric(df["ping"], errors="coerce")

    logging.info("Updating existing endpoints")
    oldest_idx = (
        df.sort_values(by="last_check_date")
        .head(args.existing_limit)
        .index
    )
    df.loc[oldest_idx] = update_existing(
        api, df.loc[oldest_idx].copy(), args.existing_limit
    )
    logging.info("Finished updating existing endpoints")

    logging.info("Searching for new endpoints")
    df = find_new(api, df, args.limit)
    logging.info("Finished searching for new endpoints")

    df = df[COLUMNS]
    df["ping"] = pd.to_numeric(df["ping"], errors="coerce")
    df.to_csv(CSV_PATH, index=False)


if __name__ == "__main__":
    main()
