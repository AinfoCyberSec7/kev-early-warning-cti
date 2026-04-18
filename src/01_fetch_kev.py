import os
import pandas as pd

KEV_URL = "https://raw.githubusercontent.com/cisagov/kev-data/main/known_exploited_vulnerabilities.csv"
OUT_DIR = "raw/kev"
OUT_PATH = os.path.join(OUT_DIR, "kev.csv")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    kev = pd.read_csv(KEV_URL)
    kev["cveID"] = kev["cveID"].astype(str).str.strip()
    kev["dateAdded"] = pd.to_datetime(kev["dateAdded"], utc=True, errors="raise")

    kev.to_csv(OUT_PATH, index=False)

    print(f"[OK] Saved: {OUT_PATH}")
    print(f"[INFO] Rows: {len(kev):,} | Columns: {len(kev.columns)}")
    print("[INFO] dateAdded range:", kev["dateAdded"].min(), "→", kev["dateAdded"].max())
    print("[INFO] Sample CVEs:", kev["cveID"].head(5).tolist())

if __name__ == "__main__":
    main()
