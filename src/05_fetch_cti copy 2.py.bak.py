import os, re, hashlib, time
import yaml
import feedparser
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from tqdm import tqdm

CFG_PATH = "config/feeds.yml"
OUT_PATH = "processed/cti_docs.parquet"

MAX_ITEMS_PER_FEED = 200
REQUEST_TIMEOUT = 30
SLEEP = 0.4

def utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def normalize_text(html_or_text: str) -> str:
    soup = BeautifulSoup(html_or_text or "", "lxml")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def stable_doc_id(source: str, url: str, title: str) -> str:
    s = f"{source}|{url}|{title}".encode("utf-8", errors="ignore")
    return hashlib.sha256(s).hexdigest()[:24]

def parse_entry_time(entry) -> str:
    for key in ("published_parsed", "updated_parsed"):
        t = getattr(entry, key, None)
        if t:
            dt = datetime(*t[:6], tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return utc_now_iso()

def fetch_article_html(url: str) -> str:
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception:
        return ""

def main():
    os.makedirs("processed", exist_ok=True)

    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sources = cfg.get("sources", [])
    if not sources:
        raise SystemExit("[BLOCKER] config/feeds.yml has no sources.")

    rows = []
    retrieved_at = utc_now_iso()

    for src in sources:
        name = src["name"]
        url = src["url"]
        ftype = src.get("type", "rss")
        stage = src.get("stage", "pre")

        if ftype != "rss":
            print(f"[WARN] Skipping non-rss source: {name}")
            continue

        print(f"[INFO] Feed: {name} -> {url}")
        feed = feedparser.parse(url)

        entries = feed.entries[:MAX_ITEMS_PER_FEED]
        for entry in tqdm(entries, desc=f"Parsing {name}", leave=False):
            title = getattr(entry, "title", "") or ""
            link = getattr(entry, "link", "") or ""
            ts = parse_entry_time(entry)

            text_parts = []
            if hasattr(entry, "summary"):
                text_parts.append(entry.summary)
            if hasattr(entry, "content") and entry.content:
                for c in entry.content:
                    text_parts.append(c.get("value", ""))

            combined = "\n".join([p for p in text_parts if p])

            # If feed content is short, try fetching full article HTML
            if len(normalize_text(combined)) < 300 and link:
                combined = combined + "\n" + fetch_article_html(link)

            norm = normalize_text(combined)

            if len(norm) < 200:
                continue

            doc_id = stable_doc_id(name, link, title)

            rows.append({
                "doc_id": doc_id,
                "timestamp_utc": ts,
                "retrieved_at_utc": retrieved_at,
                "source": name,
                "url": link,
                "title": title,
                "stage": stage,
                "normalized_text": norm
            })

            time.sleep(SLEEP)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("[BLOCKER] No CTI docs collected. Check feeds and connectivity.")

    df = df.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).reset_index(drop=True)

    df.to_parquet(OUT_PATH, index=False)
    print(f"[OK] Saved: {OUT_PATH} | Rows={len(df):,}")
    print(df[["source", "timestamp_utc", "title"]].head(3))

if __name__ == "__main__":
    main()
