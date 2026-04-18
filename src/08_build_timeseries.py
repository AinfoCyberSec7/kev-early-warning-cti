import os
import pandas as pd

IN_BACKBONE = "processed/backbone.parquet"
IN_CTI_SUM = "processed/cti_cve_summary_pre.parquet"
OUT_ENRICHED = "processed/backbone_enriched.parquet"

def main():
    os.makedirs("processed", exist_ok=True)

    bb = pd.read_parquet(IN_BACKBONE)
    cti = pd.read_parquet(IN_CTI_SUM)

    # Parse times
    bb["published_date"] = pd.to_datetime(bb["published_date"], utc=True, errors="coerce")
    bb["kev_date_added"] = pd.to_datetime(bb["kev_date_added"], utc=True, errors="coerce")

    if not cti.empty:
        cti["cti_first_seen"] = pd.to_datetime(cti["cti_first_seen"], utc=True, errors="coerce")

    enriched = bb.merge(cti, on="cve_id", how="left")

    # Lead-time features
    enriched["cti_has_signal"] = enriched["cti_first_seen"].notna().astype(int)

    # --- Date-level lead time (robust against time-of-day artifacts) ---
    enriched["cti_first_seen_date"] = enriched["cti_first_seen"].dt.floor("D")
    enriched["kev_date_added_date"] = enriched["kev_date_added"].dt.floor("D")

    enriched["lead_days_cti_to_kev"] = (
        (enriched["kev_date_added_date"] - enriched["cti_first_seen_date"])
        .dt.days
    )

    # Early-warning indicator (strictly earlier date than KEV)
    enriched["cti_before_kev"] = (
        (enriched["label_exploited"] == 1) & (enriched["lead_days_cti_to_kev"] > 0)
    ).astype(int)

    # Same-day signal (useful for analysis)
    enriched["cti_same_day_as_kev"] = (
        (enriched["label_exploited"] == 1) & (enriched["lead_days_cti_to_kev"] == 0)
    ).astype(int)

    # Days from NVD published to CTI first-seen (negative means CTI happened after NVD publish)
    enriched["delta_days_cti_minus_nvdpub"] = (
        (enriched["cti_first_seen"] - enriched["published_date"]).dt.total_seconds() / 86400.0
    )

    # Flags
    enriched["cti_before_kev"] = ((enriched["lead_days_cti_to_kev"] > 0) & (enriched["label_exploited"] == 1)).astype(int)

    enriched.to_parquet(OUT_ENRICHED, index=False)

    print(f"[OK] Saved: {OUT_ENRICHED} | Rows={len(enriched):,}")
    print("[INFO] CTI coverage:", int(enriched["cti_has_signal"].sum()), "/", len(enriched))
    print("[INFO] KEV positives with CTI before KEV:", int(enriched["cti_before_kev"].sum()))
    print("[INFO] lead_days_cti_to_kev (KEV positives only) describe:")
    pos = enriched[enriched["label_exploited"] == 1]["lead_days_cti_to_kev"].dropna()
    if len(pos):
        print(pos.describe(percentiles=[0.1,0.25,0.5,0.75,0.9]))
    else:
        print("No CTI lead-time values for positives (yet).")

if __name__ == "__main__":
    main()
