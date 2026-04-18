import os, re, hashlib, json
import pandas as pd
from datetime import datetime, timezone

# Path where you cloned the repo (sparse checkout includes advisories/)
REPO_DIR = "raw/cti/github_advisory_db/advisory-database/advisories"
OUT_PATH = "processed/cti_docs_github_advisories.parquet"

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)

def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def make_doc_id(source: str, path: str) -> str:
    s = f"{source}|{path}".encode("utf-8", errors="ignore")
    return hashlib.sha256(s).hexdigest()[:24]

def main():
    if not os.path.isdir(REPO_DIR):
        raise SystemExit(f"[BLOCKER] Missing repo dir: {REPO_DIR}\n"
                         f"Check you cloned into raw/cti/github_advisory_db/advisory-database")

    os.makedirs("processed", exist_ok=True)

    retrieved = utc_now_iso()
    source_name = "GitHub Advisory Database"

    rows = []
    for root, _, files in os.walk(REPO_DIR):
        for fn in files:
            if not fn.endswith(".json"):
                continue

            path = os.path.join(root, fn)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    j = json.load(f)
            except Exception:
                continue

            published = j.get("published")
            modified = j.get("modified")
            ts = published or modified or retrieved

            title = j.get("summary") or j.get("id") or os.path.basename(path)
            details = j.get("details") or ""
            aliases = j.get("aliases") or []

            # Text used to find CVEs + create a compact "document"
            text_blob = f"{title}\n{details}\n" + "\n".join([str(a) for a in aliases])

            cves = sorted(set(m.group(0).upper() for m in CVE_RE.finditer(text_blob)))
            if not cves:
                continue

            norm = text_blob.strip()
            if len(norm) > 5000:
                norm = norm[:5000] + "..."

            rows.append({
                "doc_id": make_doc_id(source_name, path),
                "timestamp_utc": ts,
                "retrieved_at_utc": retrieved,
                "source": source_name,
                "stage": "pre",
                "url": "",
                "title": title,
                "normalized_text": norm
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[BLOCKER] No advisory docs produced (no CVE mentions found).")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).drop_duplicates(subset=["doc_id"]).reset_index(drop=True)

    df.to_parquet(OUT_PATH, index=False)
    print(f"[OK] Saved: {OUT_PATH} | Rows={len(df):,}")
    print("time range:", df["timestamp_utc"].min(), "->", df["timestamp_utc"].max())
    print(df[["source","stage","timestamp_utc","title"]].head(3))

if __name__ == "__main__":
    main()
