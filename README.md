Basic script to converse with Ollama endpoints.
"endpoints.csv" needs to be populated with valid Ollama endpoints.

## Automatic population

The `shodan_scan.py` helper uses the [Shodan](https://www.shodan.io/) API to
keep `endpoints.csv` up to date. It performs two tasks:

1. Verify the online status of servers already present in the CSV using
   batched queries per port.
2. Search Shodan for additional public Ollama instances and append them.

By default the script looks for hosts on the standard Ollama port (11434)
whose HTTP response contains the text "Ollama is running".

Create a `config.json` next to `shodan_scan.py` with your API key. A
`config.example.json` template is provided:

```json
{
  "SHODAN_API_KEY": "your_key_here"
}
```

Alternatively, set the `SHODAN_API_KEY` environment variable. The file takes
precedence over the environment variable if both are present. Run the script:

```bash
python shodan_scan.py
```

The script uses Python's built-in logging module to report its progress. By
default it logs informational messages; pass `--verbose` to enable debug level
output:

```bash
python shodan_scan.py --verbose
```

To control API usage, results from each Shodan query are limited to 100 entries by default.
Use `--limit` to change how many new endpoints are fetched per query.
Existing entries are checked in batches of 100; adjust this with `--existing-limit` to
control API usage for large CSVs.

```bash
python shodan_scan.py --limit 50 --existing-limit 25
```

The script requires the `shodan` and `pandas` packages. It also enriches each
entry with metadata provided by Shodan such as hostnames, organisation, ISP and
geolocation (city, region, country and coordinates) so the CSV remains as
informative as possible.
