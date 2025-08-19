import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import shodan

CSV_PATH = "endpoints.csv"
CONFIG_PATH = Path(__file__).with_name("config.json")
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
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


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
                results = api.search(query, limit=batch_size).get("matches", [])
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
            df.at[idx, "org"] = r.get("org", "")
            df.at[idx, "isp"] = r.get("isp", "")
            df.at[idx, "city"] = location.get("city", "")
            df.at[idx, "region"] = location.get("region_code") or location.get("region_name", "")
            df.at[idx, "country"] = location.get("country_name", "")
            df.at[idx, "latitude"] = location.get("latitude", "")
            df.at[idx, "longitude"] = location.get("longitude", "")
        else:
            df.at[idx, "is_active"] = False
            df.at[idx, "inactive_reason"] = errors.get(key, "port closed")
            df.at[idx, "last_check_date"] = now
    return df


def find_new(api, df, limit):
    existing = set(zip(df["ip"], df["port"]))
    new_rows = []
    for query in SHODAN_QUERIES:
        logging.info(f"Executing Shodan query: {query} (limit {limit})")
        try:
            results = api.search(query, limit=limit).get("matches", [])
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
            new_rows.append({
                "id": f"Shodan {ip}:{port}",
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
                "org": r.get("org", ""),
                "isp": r.get("isp", ""),
                "city": location.get("city", ""),
                "region": location.get("region_code") or location.get("region_name", ""),
                "country": location.get("country_name", ""),
                "latitude": location.get("latitude", ""),
                "longitude": location.get("longitude", ""),
            })
            existing.add((ip, str(port)))
            count += 1
        logging.info(f"Processed {count} results for query: {query}")
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
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
        default=100,
        help="Maximum results to fetch per Shodan query",
    )
    parser.add_argument(
        "--existing-limit",
        type=int,
        default=100,
        help="Maximum endpoints to verify per query when checking existing entries",
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

    logging.info("Updating existing endpoints")
    df = update_existing(api, df, args.existing_limit)
    logging.info("Finished updating existing endpoints")

    logging.info("Searching for new endpoints")
    df = find_new(api, df, args.limit)
    logging.info("Finished searching for new endpoints")

    df = df[COLUMNS]
    df.to_csv(CSV_PATH, index=False)


if __name__ == "__main__":
    main()
