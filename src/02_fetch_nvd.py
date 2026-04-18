import os, time, json, math
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta, timezone

API = "https://services.nvd.nist.gov/rest/json/cves/2.0"

OUT_DIR = "raw/nvd"
OUT_JSONL = os.path.join(OUT_DIR, "nvd_cves.jsonl")
OUT_FLAT = "processed/nvd_flat.parquet"

PUB_START = "2021-01-01T00:00:00.000Z"
PUB_END   = "2025-12-31T23:59:59.999Z"

WINDOW_DAYS = 90          # <= keep windows modest (30–90 is safe)
RESULTS_PER_PAGE = 2000   # max page size
SLEEP_KEY = 0.8
SLEEP_NO_KEY = 2.0

def iso(dtobj):
    # NVD expects Zulu time; keep milliseconds
    return dtobj.strftime("%Y-%m-%dT%H:%M:%S.000Z")

def parse_iso_z(s):
    # accepts strings like 2026-02-21T20:55:27.411
    # but we only need our own windows here
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def extract_rows(page_json):
    rows = []
    for item in page_json.get("vulnerabilities", []):
        cve = item.get("cve", {})
        cve_id = cve.get("id")
        published = cve.get("published")
        last_modified = cve.get("lastModified")

        # English description
        desc = ""
        for d in cve.get("descriptions", []):
            if d.get("lang") == "en":
                desc = d.get("value", "")
                break
        if not desc and cve.get("descriptions"):
            desc = cve["descriptions"][0].get("value", "")

        # CWE
        cwe = None
        weaknesses = cve.get("weaknesses", [])
        if weaknesses:
            for wdesc in weaknesses[0].get("description", []):
                if wdesc.get("lang") == "en":
                    cwe = wdesc.get("value")
                    break

        # CVSS v3.x (use first metric record)
        cvss_base = None
        cvss_vector = None
        metrics = cve.get("metrics", {})
        for key in ("cvssMetricV31", "cvssMetricV30"):
            if key in metrics and metrics[key]:
                cvss = metrics[key][0].get("cvssData", {})
                cvss_base = cvss.get("baseScore")
                cvss_vector = cvss.get("vectorString")
                break

        rows.append({
            "cve_id": cve_id,
            "published_date": published,
            "last_modified_date": last_modified,
            "description_en": desc,
            "cwe_id": cwe,
            "cvss_base": cvss_base,
            "cvss_vector": cvss_vector,
        })
    return rows

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("processed", exist_ok=True)

    api_key = os.getenv("NVD_API_KEY", "")
    headers = {"apiKey": api_key} if api_key else {}

    # Convert bounds to datetime
    start_dt = parse_iso_z(PUB_START)
    end_dt = parse_iso_z(PUB_END)

    # Reset output jsonl for a fresh run
    open(OUT_JSONL, "w").close()

    all_rows = []
    current = start_dt

    window_idx = 0
    while current < end_dt:
        window_idx += 1
        window_end = min(current + timedelta(days=WINDOW_DAYS), end_dt)

        pub_start = iso(current)
        pub_end = iso(window_end)

        # First call in the window to get totalResults
        params = {
            "pubStartDate": pub_start,
            "pubEndDate": pub_end,
            "resultsPerPage": RESULTS_PER_PAGE,
            "startIndex": 0,
        }

        r = requests.get(API, params=params, headers=headers, timeout=60)
        if r.status_code == 404:
            raise SystemExit(
                f"[BLOCKER] 404 for window {pub_start} → {pub_end}. "
                "Reduce WINDOW_DAYS (e.g., 30) and retry."
            )
        r.raise_for_status()
        first = r.json()

        total = first.get("totalResults", 0)
        pages = math.ceil(total / RESULTS_PER_PAGE) if total else 0

        print(f"[INFO] Window {window_idx}: {pub_start} → {pub_end} | total={total:,} | pages≈{pages}")

        # Save + process first page
        with open(OUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(first) + "\n")
        all_rows.extend(extract_rows(first))

        # Remaining pages for this window
        start_index = RESULTS_PER_PAGE
        for _ in tqdm(range(1, pages), desc=f"Window {window_idx} pages", leave=False):
            params["startIndex"] = start_index
            r = requests.get(API, params=params, headers=headers, timeout=60)
            r.raise_for_status()
            page = r.json()

            with open(OUT_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(page) + "\n")
            all_rows.extend(extract_rows(page))

            start_index += RESULTS_PER_PAGE
            time.sleep(SLEEP_KEY if api_key else SLEEP_NO_KEY)

        # Move to next window
        current = window_end
        time.sleep(SLEEP_KEY if api_key else SLEEP_NO_KEY)

    df = pd.DataFrame(all_rows)

    # Normalize timestamps and de-duplicate CVEs (safe if overlaps happen)
    df["published_date"] = pd.to_datetime(df["published_date"], utc=True, errors="coerce")
    df["last_modified_date"] = pd.to_datetime(df["last_modified_date"], utc=True, errors="coerce")
    df = df.drop_duplicates(subset=["cve_id"]).reset_index(drop=True)

    df.to_parquet(OUT_FLAT, index=False)
    print(f"[OK] Saved: {OUT_FLAT} | Rows={len(df):,}")
    print("[INFO] published_date nulls:", int(df["published_date"].isna().sum()))
    print("[INFO] CVSS base non-null:", int(df["cvss_base"].notna().sum()))
    print("[INFO] Unique CVSS vectors (non-null):", int(df["cvss_vector"].nunique(dropna=True)))

if __name__ == "__main__":
    main()
