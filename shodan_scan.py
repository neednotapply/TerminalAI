import os
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import shodan

CSV_PATH = "endpoints.csv"
CONFIG_PATH = Path(__file__).with_name("config.json")
SHODAN_QUERIES = [
    'port:11434 "Ollama"'
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
    try:
        with CONFIG_PATH.open() as f:
            key = json.load(f).get("SHODAN_API_KEY")
    except (OSError, json.JSONDecodeError):
        key = None
    return key or os.getenv("SHODAN_API_KEY")


def update_existing(api, df):
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
        ip_query = " OR ".join(f"ip:{ip}" for ip in ips)
        query = f"port:{port} ({ip_query})"
        try:
            results = api.search(query, limit=len(ips)).get("matches", [])
            for r in results:
                key = (r.get("ip_str"), r.get("port"))
                details[key] = r
        except shodan.APIError as e:
            for ip in ips:
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


def find_new(api, df):
    existing = set(zip(df["ip"], df["port"]))
    new_rows = []
    for query in SHODAN_QUERIES:
        try:
            results = api.search(query).get("matches", [])
        except shodan.APIError:
            continue
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
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def main():
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
    df = update_existing(api, df)
    df = find_new(api, df)
    df = df[COLUMNS]
    df.to_csv(CSV_PATH, index=False)


if __name__ == "__main__":
    main()
