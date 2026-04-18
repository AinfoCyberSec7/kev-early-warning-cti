import pandas as pd

A = "processed/cti_docs.parquet"                      # RSS CTI
B = "processed/cti_docs_github_advisories.parquet"    # Archival advisories
OUT = "processed/cti_docs_merged.parquet"

def main():
    a = pd.read_parquet(A)
    b = pd.read_parquet(B)

    df = pd.concat([a, b], ignore_index=True)
    df = df.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"]).reset_index(drop=True)

    df.to_parquet(OUT, index=False)
    print(f"[OK] Saved: {OUT} | Rows={len(df):,}")
    print("stage counts:\n", df["stage"].value_counts())
    print("time range:", df["timestamp_utc"].min(), "->", df["timestamp_utc"].max())

if __name__ == "__main__":
    main()
