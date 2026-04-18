import os, time
import requests
import pandas as pd
from tqdm import tqdm

EPSS_API = "https://api.first.org/data/v1/epss"

IN_NVD = "processed/nvd_flat.parquet"
OUT_DIR = "raw/epss"
OUT_PATH = os.path.join(OUT_DIR, "epss.csv")

CHUNK = 200      # safe chunk size for query length
SLEEP = 0.35     # be polite to API

def fetch_chunk(cves):
    params = {"cve": ",".join(cves)}
    r = requests.get(EPSS_API, params=params, timeout=60)
    r.raise_for_status()
    return r.json().get("data", [])

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    nvd = pd.read_parquet(IN_NVD)
    cves = nvd["cve_id"].dropna().astype(str).unique().tolist()
    print(f"[INFO] CVEs for EPSS: {len(cves):,}")

    rows = []
    for i in tqdm(range(0, len(cves), CHUNK), desc="Fetching EPSS"):
        rows.extend(fetch_chunk(cves[i:i+CHUNK]))
        time.sleep(SLEEP)

    epss = pd.DataFrame(rows)
    epss.to_csv(OUT_PATH, index=False)

    print(f"[OK] Saved: {OUT_PATH} | Rows={len(epss):,} | Cols={len(epss.columns)}")
    if len(epss):
        print("[INFO] Columns:", epss.columns.tolist())
        print("[INFO] Sample:", epss.head(3).to_dict(orient="records"))

if __name__ == "__main__":
    main()
