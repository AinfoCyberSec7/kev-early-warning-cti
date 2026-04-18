import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

IN_PATH = "processed/backbone_enriched.parquet"
OUT_DIR = "reports/paper_assets"
OUT_CSV = os.path.join(OUT_DIR, "table4_ablation_results.csv")

# Budgets for recall@K (alert budgets)
K_LIST = [100, 250, 500, 1000, 2500, 5000]

# Time-aware split (publication year)
TRAIN_YEARS = (2021, 2023)   # inclusive
TEST_YEARS  = (2024, 2025)   # inclusive

def pick_col(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def safe_to_datetime(s):
    return pd.to_datetime(s, utc=True, errors="coerce")

def compute_recalls(y_true, scores, k):
    """
    Returns recall@k for positive class.
    """
    k = min(k, len(scores))
    idx = np.argsort(-scores)[:k]
    return float(y_true.iloc[idx].sum() / max(1, int(y_true.sum())))

def compute_leadtime_aware_recall(y_true, scores, lead_days, k):
    """
    Lead-time-aware recall@k among early-warning eligible positives:
    denominator = count of positives with lead_days > 0
    numerator = count of those in top-k
    """
    eligible = (y_true == 1) & (lead_days > 0)
    denom = int(eligible.sum())
    if denom == 0:
        return np.nan
    k = min(k, len(scores))
    idx = np.argsort(-scores)[:k]
    num = int(eligible.iloc[idx].sum())
    return float(num / denom)

def build_features(df):
    """
    Construct leakage-aware CTI features at prediction time.
    Prediction time is approximated as CVE published_date (disclosure-time features).
    We only allow CTI features if cti_first_seen <= published_date.
    This is a conservative 'early data' setting.
    """
    cols = df.columns

    y_col = pick_col(cols, ["label_exploited"])
    pub_col = pick_col(cols, ["published_date", "published", "cve_published", "publishedDate"])
    cti_first = pick_col(cols, ["cti_first_seen"])
    cti_mentions = pick_col(cols, ["cti_mentions"])
    cti_sources = pick_col(cols, ["cti_sources"])
    lead_col = pick_col(cols, ["lead_days_cti_to_kev"])

    epss_score = pick_col(cols, ["epss_score", "epss"])
    epss_pct = pick_col(cols, ["epss_percentile", "epss_percentile_score", "percentile"])

    # NVD / CVSS / CWE candidates (names can vary)
    cvss_col = None
    for c in cols:
        cl = c.lower()
        if "cvss" in cl and ("base" in cl or "score" in cl):
            cvss_col = c
            break

    cwe_col = pick_col(cols, ["cwe", "cwe_id", "cwe_primary", "primary_cwe"])

    if y_col is None:
        raise SystemExit("[BLOCKER] Missing label column (expected label_exploited).")
    if pub_col is None:
        raise SystemExit("[BLOCKER] Missing published_date column from backbone.")
    if epss_score is None:
        raise SystemExit("[BLOCKER] Missing EPSS score column (expected epss_score).")
    if lead_col is None:
        # Not fatal for classification, but needed for lead-time-aware metrics
        print("[WARN] Missing lead_days_cti_to_kev; lead-time-aware metrics will be NaN.")
    if cti_first is None:
        print("[WARN] Missing cti_first_seen; CTI ablation will degrade to zeros.")

    # Parse dates
    df = df.copy()
    df[pub_col] = safe_to_datetime(df[pub_col])

    if cti_first is not None:
        df[cti_first] = safe_to_datetime(df[cti_first])
        # CTI available at prediction time only if first_seen <= published_date
        cti_available = (df[cti_first].notna()) & (df[pub_col].notna()) & (df[cti_first] <= df[pub_col])
    else:
        cti_available = pd.Series(False, index=df.index)

    # Leakage-aware CTI features
    df["cti_has_signal_at_pub"] = cti_available.astype(int)

    if cti_mentions is not None:
        df["cti_mentions_at_pub"] = np.where(cti_available, df[cti_mentions].fillna(0), 0)
    else:
        df["cti_mentions_at_pub"] = 0

    if cti_sources is not None:
        df["cti_sources_at_pub"] = np.where(cti_available, df[cti_sources].fillna(0), 0)
    else:
        df["cti_sources_at_pub"] = 0

    # recency feature: days since CTI first seen at publication time (only if available)
    if cti_first is not None:
        dt_days = (df[pub_col] - df[cti_first]).dt.total_seconds() / 86400.0
        df["days_since_cti_first_seen_at_pub"] = np.where(cti_available, dt_days.clip(lower=0), np.nan)
    else:
        df["days_since_cti_first_seen_at_pub"] = np.nan

    # Core feature columns for each ablation
    feat = {
        "A0_EPSS_only": {
            "numeric": [epss_score, epss_pct],
            "categorical": [],
        },
        "A1_EPSS_NVD": {
            "numeric": [epss_score, epss_pct] + ([cvss_col] if cvss_col else []),
            "categorical": ([cwe_col] if cwe_col else []),
        },
        "A2_EPSS_CTIpre": {
            "numeric": [epss_score, epss_pct, "cti_has_signal_at_pub", "cti_mentions_at_pub", "cti_sources_at_pub", "days_since_cti_first_seen_at_pub"],
            "categorical": [],
        },
        "A3_Hybrid": {
            "numeric": [epss_score, epss_pct] + ([cvss_col] if cvss_col else []) + ["cti_has_signal_at_pub", "cti_mentions_at_pub", "cti_sources_at_pub", "days_since_cti_first_seen_at_pub"],
            "categorical": ([cwe_col] if cwe_col else []),
        },
    }

    return df, feat, y_col, pub_col, lead_col, cvss_col, cwe_col

def make_model(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop"
    )

    base = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )

    # Calibrated wrapper improves probability quality (helpful for thresholding)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    return Pipeline(steps=[("pre", pre), ("clf", clf)])

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(IN_PATH)

    df, feat, y_col, pub_col, lead_col, cvss_col, cwe_col = build_features(df)

    # Build time-aware split
    df["pub_year"] = df[pub_col].dt.year
    train = df[(df["pub_year"] >= TRAIN_YEARS[0]) & (df["pub_year"] <= TRAIN_YEARS[1])].copy()
    test  = df[(df["pub_year"] >= TEST_YEARS[0]) & (df["pub_year"] <= TEST_YEARS[1])].copy()

    # Drop rows with missing pub_date (cannot place in time split)
    train = train[train[pub_col].notna()]
    test = test[test[pub_col].notna()]

    if len(train) == 0 or len(test) == 0:
        raise SystemExit("[BLOCKER] Empty train/test after time split. Check published_date parsing.")

    y_train = train[y_col].astype(int)
    y_test = test[y_col].astype(int)

    if lead_col and lead_col in test.columns:
        lead_test = pd.to_numeric(test[lead_col], errors="coerce")
    else:
        lead_test = pd.Series(np.nan, index=test.index)

    results = []

    for name, spec in feat.items():
        numeric_cols = [c for c in spec["numeric"] if c is not None]
        categorical_cols = [c for c in spec["categorical"] if c is not None]

        # Ensure feature columns exist
        missing = [c for c in numeric_cols + categorical_cols if c not in train.columns]
        if missing:
            print(f"[WARN] {name}: missing columns {missing}. Skipping.")
            continue

        model = make_model(numeric_cols, categorical_cols)
        model.fit(train[numeric_cols + categorical_cols], y_train)

        # Probabilities for class 1
        p = model.predict_proba(test[numeric_cols + categorical_cols])[:, 1]

        ap = float(average_precision_score(y_test, p))
        try:
            ra = float(roc_auc_score(y_test, p))
        except Exception:
            ra = np.nan

        brier = float(brier_score_loss(y_test, p))

        row = {
            "Model": name,
            "TrainYears": f"{TRAIN_YEARS[0]}-{TRAIN_YEARS[1]}",
            "TestYears": f"{TEST_YEARS[0]}-{TEST_YEARS[1]}",
            "PR-AUC(AP)": ap,
            "ROC-AUC": ra,
            "Brier": brier,
            "Test_N": int(len(test)),
            "Test_Positives": int(y_test.sum()),
        }

        # Recall@K budgets
        for k in K_LIST:
            row[f"Recall@{k}"] = compute_recalls(y_test, pd.Series(p, index=y_test.index), k)
            row[f"EW_Recall@{k}"] = compute_leadtime_aware_recall(y_test, pd.Series(p, index=y_test.index), lead_test, k)

        results.append(row)

    out = pd.DataFrame(results).sort_values("PR-AUC(AP)", ascending=False)
    out.to_csv(OUT_CSV, index=False)

    print("[OK] Wrote:", OUT_CSV)
    print(out.head(10).to_string(index=False))

    # Also print which CVSS/CWE columns were used (useful for methods section)
    print("\n[INFO] Detected columns:")
    print(" - label:", y_col)
    print(" - published_date:", pub_col)
    print(" - cvss:", cvss_col)
    print(" - cwe:", cwe_col)

if __name__ == "__main__":
    main()
