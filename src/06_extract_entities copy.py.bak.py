import os, re
import pandas as pd

IN_CTI = "processed/cti_docs.parquet"
IN_KEV = "raw/kev/kev.csv"
OUT_MENTIONS = "processed/entity_mentions.parquet"
OUT_CVE_SUMMARY = "processed/cti_cve_summary.parquet"

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
CWE_RE = re.compile(r"\bCWE-\d+\b", re.IGNORECASE)

def build_vendor_product_lexicon(kev: pd.DataFrame):
    """
    Conservative lexicon from KEV vendorProject/product.
    We only keep reasonably distinctive vendor names to avoid false positives.
    """
    kev = kev.copy()
    kev["vendorProject"] = kev["vendorProject"].fillna("").astype(str).str.strip()
    kev["product"] = kev["product"].fillna("").astype(str).str.strip()

    vendors = sorted(set(v for v in kev["vendorProject"].unique().tolist() if v))
    products = sorted(set(p for p in kev["product"].unique().tolist() if p))

    # Filter: avoid extremely short or common tokens
    def ok_token(x: str) -> bool:
        x2 = x.strip()
        if len(x2) < 4:
            return False
        # drop very generic names that create noise
        bad = {"windows", "linux", "server", "client", "router", "switch", "web", "device"}
        if x2.lower() in bad:
            return False
        return True

    vendors = [v for v in vendors if ok_token(v)]
    products = [p for p in products if ok_token(p)]

    return vendors, products

def find_mentions(text: str, pattern: re.Pattern):
    return sorted(set(m.group(0).upper() for m in pattern.finditer(text or "")))

def main():
    os.makedirs("processed", exist_ok=True)

    cti = pd.read_parquet(IN_CTI)
    kev = pd.read_csv(IN_KEV)

    # Parse CTI timestamps robustly
    cti["timestamp_utc"] = pd.to_datetime(cti["timestamp_utc"], utc=True, errors="coerce")
    cti = cti.dropna(subset=["timestamp_utc"]).reset_index(drop=True)

    vendors, products = build_vendor_product_lexicon(kev)

    # Compile vendor/product regexes (case-insensitive, word-boundary-ish)
    vendor_res = [(v, re.compile(rf"(?i)\b{re.escape(v)}\b")) for v in vendors]
    product_res = [(p, re.compile(rf"(?i)\b{re.escape(p)}\b")) for p in products]

    rows = []

    for _, r in cti.iterrows():
        doc_id = r["doc_id"]
        ts = r["timestamp_utc"]
        src = r["source"]
        url = r.get("url", "")
        title = r.get("title", "")
        text = (r.get("title", "") or "") + "\n" + (r.get("normalized_text", "") or "")

        # CVE mentions
        for cve in find_mentions(text, CVE_RE):
            rows.append({
                "doc_id": doc_id, "timestamp_utc": ts, "source": src, "url": url, "title": title,
                "entity_type": "cve", "entity_value": cve
            })

        # CWE mentions
        for cwe in find_mentions(text, CWE_RE):
            rows.append({
                "doc_id": doc_id, "timestamp_utc": ts, "source": src, "url": url, "title": title,
                "entity_type": "cwe", "entity_value": cwe.upper()
            })

        # Vendor mentions (conservative)
        for v, rx in vendor_res:
            if rx.search(text or ""):
                rows.append({
                    "doc_id": doc_id, "timestamp_utc": ts, "source": src, "url": url, "title": title,
                    "entity_type": "vendor", "entity_value": v
                })

        # Product mentions (conservative)
        for p, rx in product_res:
            if rx.search(text or ""):
                rows.append({
                    "doc_id": doc_id, "timestamp_utc": ts, "source": src, "url": url, "title": title,
                    "entity_type": "product", "entity_value": p
                })

    mentions = pd.DataFrame(rows)
    if mentions.empty:
        raise SystemExit("[BLOCKER] No entity mentions extracted. Check CTI content and patterns.")

    mentions = mentions.drop_duplicates().reset_index(drop=True)
    mentions.to_parquet(OUT_MENTIONS, index=False)
    print(f"[OK] Saved: {OUT_MENTIONS} | Rows={len(mentions):,}")

    # CVE-level summary: first seen in CTI, mention count, distinct sources
    cve_m = mentions[mentions["entity_type"] == "cve"].copy()
    if not cve_m.empty:
        summary = (
            cve_m.groupby("entity_value")
            .agg(
                cti_first_seen=("timestamp_utc", "min"),
                cti_mentions=("doc_id", "count"),
                cti_sources=("source", "nunique"),
            )
            .reset_index()
            .rename(columns={"entity_value": "cve_id"})
        )
    else:
        summary = pd.DataFrame(columns=["cve_id","cti_first_seen","cti_mentions","cti_sources"])

    summary.to_parquet(OUT_CVE_SUMMARY, index=False)
    print(f"[OK] Saved: {OUT_CVE_SUMMARY} | CVEs={len(summary):,}")
    print(summary.head(5))

if __name__ == "__main__":
    main()
