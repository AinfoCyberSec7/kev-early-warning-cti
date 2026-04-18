import os
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "reports/paper2_assets"

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    bb = pd.read_parquet("processed/backbone_enriched.parquet")

    N = len(bb)
    kev_pos = int((bb["label_exploited"] == 1).sum())
    cti_cov = int(bb["cti_first_seen"].notna().sum())

    pos = bb[(bb["label_exploited"] == 1) & (bb["cti_first_seen"].notna())].copy()
    pos_n = len(pos)
    ew_n = int((pos["lead_days_cti_to_kev"] > 0).sum()) if pos_n else 0
    ew_share = (ew_n / pos_n) if pos_n else 0.0

    # Table 1: dataset summary
    t1 = pd.DataFrame([
        ["NVD slice CVEs (N)", N],
        ["KEV positives", kev_pos],
        ["RSS-only CTI coverage (cti_first_seen non-null)", cti_cov],
        ["RSS-only CTI coverage share", cti_cov / N],
        ["KEV positives with any RSS CTI signal", pos_n],
        ["KEV positives with CTI before KEV (lead_days>0)", ew_n],
        ["Early-warning share among KEV+CTI", ew_share],
    ], columns=["Metric", "Value"])
    t1.to_csv(f"{OUTDIR}/paper2_table1_rss_summary.csv", index=False)

    # Table 2: lead time stats for KEV positives with CTI
    if pos_n:
        desc = pos["lead_days_cti_to_kev"].describe(percentiles=[.1,.25,.5,.75,.9]).to_frame("lead_days").reset_index()
        desc.rename(columns={"index":"stat"}, inplace=True)
    else:
        desc = pd.DataFrame([["count", 0]], columns=["stat","lead_days"])
    desc.to_csv(f"{OUTDIR}/paper2_table2_rss_leadtime_stats.csv", index=False)

    # Table 3: early warning counts
    t3 = pd.DataFrame([
        ["KEV positives with RSS CTI signal", pos_n],
        ["KEV positives with CTI before KEV (lead_days>0)", ew_n],
        ["share_with_early_warning", ew_share],
    ], columns=["Metric","Value"])
    t3.to_csv(f"{OUTDIR}/paper2_table3_rss_earlywarning_counts.csv", index=False)

    # Figure 1: CTI coverage bar
    plt.figure()
    plt.bar(["All CVEs", "CTI-covered"], [N, cti_cov])
    plt.ylabel("Count")
    plt.title("RSS-only CTI coverage (count)")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/paper2_fig1_rss_coverage.png", dpi=200)
    plt.close()

    # Figure 2: Lead-time histogram (if any KEV positives have CTI)
    if pos_n:
        plt.figure()
        plt.hist(pos["lead_days_cti_to_kev"].dropna(), bins=20)
        plt.xlabel("Lead days (KEV dateAdded - RSS CTI first_seen)")
        plt.ylabel("KEV positives (count)")
        plt.title("RSS-only lead-time distribution for KEV positives")
        plt.tight_layout()
        plt.savefig(f"{OUTDIR}/paper2_fig2_rss_leadtime_hist.png", dpi=200)
        plt.close()

    print("[OK] Wrote RSS-only paper2 assets to:", OUTDIR)
    for f in sorted(os.listdir(OUTDIR)):
        print(" -", f)

if __name__ == "__main__":
    main()
