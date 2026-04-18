import os
import pandas as pd

IN_NVD = "processed/nvd_flat.parquet"
IN_KEV = "raw/kev/kev.csv"
IN_EPSS = "raw/epss/epss_scores-2026-02-21.csv"
EPSS_DATE = "2026-02-21"
OUT = "processed/backbone.parquet"

def main():
    os.makedirs("processed", exist_ok=True)

    nvd = pd.read_parquet(IN_NVD)
    kev = pd.read_csv(IN_KEV)
    # EPSS daily dump has a leading metadata line starting with '#'
    epss = pd.read_csv(IN_EPSS, comment="#")

    kev["cveID"] = kev["cveID"].astype(str).str.strip()
    kev["dateAdded"] = pd.to_datetime(kev["dateAdded"], utc=True, errors="raise")

    epss = epss.rename(columns={"cve": "cve_id", "epss": "epss_score", "percentile": "epss_percentile"})

    backbone = nvd.merge(
        kev[["cveID", "dateAdded"]].rename(columns={"cveID": "cve_id", "dateAdded": "kev_date_added"}),
        on="cve_id", how="left"
    )
    backbone["label_exploited"] = backbone["kev_date_added"].notna().astype(int)

    backbone = backbone.merge(
        epss[["cve_id", "epss_score", "epss_percentile"]],
        on="cve_id", how="left"
    )
    backbone["epss_date"] = EPSS_DATE

    # strict timestamp parsing check
    backbone["published_date"] = pd.to_datetime(backbone["published_date"], utc=True, errors="raise")
    backbone["last_modified_date"] = pd.to_datetime(backbone["last_modified_date"], utc=True, errors="raise")

    backbone.to_parquet(OUT, index=False)

    # Critical sanity checks
    vc = backbone["label_exploited"].value_counts()
    print(f"[OK] Saved: {OUT} | Rows={len(backbone):,}")
    print("[INFO] label_exploited distribution:\n", vc)
    if vc.shape[0] < 2:
        raise SystemExit("[BLOCKER] Only one class in labels. Check KEV join or window.")

    print("[INFO] EPSS coverage:", int(backbone["epss_score"].notna().sum()), "/", len(backbone))
    print("[INFO] CVSS coverage:", int(backbone["cvss_base"].notna().sum()), "/", len(backbone))

if __name__ == "__main__":
    main()
