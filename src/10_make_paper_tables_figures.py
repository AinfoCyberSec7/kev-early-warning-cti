import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "reports/paper_assets"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    bb = pd.read_parquet("processed/backbone_enriched.parquet")

    # --- Table 1: Dataset summary ---
    total = len(bb)
    kev_pos = int((bb["label_exploited"]==1).sum())
    kev_neg = total - kev_pos
    cti_cov = int(bb["cti_first_seen"].notna().sum())
    epss_cov = int(bb["epss_score"].notna().sum())

    # Auto-detect a CVSS base score column (names vary by NVD flattening)
    cvss_candidates = [c for c in bb.columns if "cvss" in c.lower() and ("base" in c.lower() or "score" in c.lower())]
    cvss_col = cvss_candidates[0] if cvss_candidates else None
    cvss_cov = int(bb[cvss_col].notna().sum()) if cvss_col else 0

    tbl1 = pd.DataFrame([{
       "NVD CVEs (2021-2025 slice)": total,
       "KEV positives": kev_pos,
       "KEV negatives": kev_neg,
       "CTI coverage": cti_cov,
       "EPSS coverage": epss_cov,
       "CVSS coverage": cvss_cov,
       "CVSS column used": (cvss_col or "none"),
    }])
    tbl1.to_csv(f"{OUT_DIR}/table1_dataset_summary.csv", index=False)

    # --- Table 2: Early-warning lead time stats for KEV positives with CTI ---
    pos = bb[(bb["label_exploited"]==1) & (bb["cti_first_seen"].notna())].copy()
    pos = pos.dropna(subset=["lead_days_cti_to_kev"])

    tbl2 = pos["lead_days_cti_to_kev"].describe(percentiles=[.1,.25,.5,.75,.9]).to_frame().T
    tbl2.to_csv(f"{OUT_DIR}/table2_leadtime_stats.csv", index=False)

    # --- Table 3: Count of KEV positives with pre-signal ---
    tbl3 = pd.DataFrame([{
        "KEV positives with CTI signal": len(pos),
        "KEV positives with CTI before KEV (lead_days>0)": int((pos["lead_days_cti_to_kev"]>0).sum()),
        "share_with_early_warning": float((pos["lead_days_cti_to_kev"]>0).mean())
    }])
    tbl3.to_csv(f"{OUT_DIR}/table3_earlywarning_counts.csv", index=False)

    # --- Figure 1: Lead time histogram ---
    plt.figure()
    vals = pos["lead_days_cti_to_kev"].clip(lower=-365, upper=365)  # cap for readable plot
    plt.hist(vals.dropna(), bins=60)
    plt.xlabel("Lead days (CTI first seen minus KEV dateAdded), capped [-365, 365]")
    plt.ylabel("Count (KEV positives)")
    plt.title("Distribution of CTI lead time relative to KEV inclusion")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_leadtime_hist.png", dpi=200)
    plt.close()

    # --- Figure 2: CDF of positive lead times ---
    pos_only = pos[pos["lead_days_cti_to_kev"]>0]["lead_days_cti_to_kev"].sort_values()
    if len(pos_only) > 0:
        y = np.arange(1, len(pos_only)+1) / len(pos_only)
        plt.figure()
        plt.plot(pos_only.values, y)
        plt.xlabel("Lead days (positive only)")
        plt.ylabel("CDF")
        plt.title("CDF of positive early-warning lead times (KEV positives)")
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/fig2_positive_leadtime_cdf.png", dpi=200)
        plt.close()

    print("[OK] Wrote tables/figures to:", OUT_DIR)
    print("Examples:")
    print(" -", f"{OUT_DIR}/table1_dataset_summary.csv")
    print(" -", f"{OUT_DIR}/fig1_leadtime_hist.png")

if __name__ == "__main__":
    main()
