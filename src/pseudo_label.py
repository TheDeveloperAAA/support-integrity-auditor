"""
Phase 2b — Pseudo-label generation (self-supervised supervision signal).

Pipeline:
  1. compute the 3 Stage-1 severity signals (src/signals.py)
  2. fuse them -> inferred_severity (label-independent)
  3. severity_delta = inferred_severity - assigned_priority_score
  4. mismatch = 1 if |severity_delta| >= TAU   (TAU auto-calibrated on TRAIN only)
     direction:  delta>0 -> Hidden Crisis (under-prioritized)
                 delta<0 -> False Alarm   (over-prioritized)
  5. emit ablation (each signal alone vs fused) + pairwise signal agreement.

Run:  python3 src/pseudo_label.py
Out:  artifacts/data/pseudo_labeled.parquet
      artifacts/metrics/pseudolabel_stats.json
      artifacts/metrics/ablation.csv
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import signals as S

# --------------------------------------------------------------------------- #
def fuse(df: pd.DataFrame) -> pd.Series:
    w = C.SIGNAL_WEIGHTS
    return (w["text_rule"] * df["sev_text"]
            + w["embedding"] * df["sev_embed"]
            + w["resolution_time"] * df["sev_rt"])

def direction(delta: np.ndarray, tau: float) -> np.ndarray:
    """+1 hidden crisis, -1 false alarm, 0 consistent."""
    d = np.zeros(len(delta), dtype=int)
    d[delta >= tau] = 1
    d[delta <= -tau] = -1
    return d

def calibrate_tau(delta_train: np.ndarray) -> float:
    """Pick TAU so the TRAIN positive rate lands in TARGET_POSITIVE_RATE,
    closest to the band midpoint."""
    lo, hi = C.TARGET_POSITIVE_RATE
    mid = (lo + hi) / 2
    best, best_gap = C.MISMATCH_TAU, 1e9
    for tau in np.arange(0.15, 0.50, 0.005):
        rate = float((np.abs(delta_train) >= tau).mean())
        gap = abs(rate - mid)
        if lo <= rate <= hi and gap < best_gap:
            best, best_gap = float(tau), gap
    return best

# --------------------------------------------------------------------------- #
def single_signal_labels(df, sev_col, tau):
    delta = df[sev_col].values - df["priority_score"].values
    return (np.abs(delta) >= tau).astype(int), direction(delta, tau)

def build_ablation(df, tau) -> pd.DataFrame:
    rows = []
    fused_mis = df["mismatch"].values
    for name, col in [("text_rule", "sev_text"), ("embedding", "sev_embed"),
                      ("resolution_time", "sev_rt"), ("FUSED", "inferred_severity")]:
        mis, dirn = single_signal_labels(df, col, tau)
        rows.append({
            "signal": name,
            "weight": C.SIGNAL_WEIGHTS.get(name, 1.0 if name == "FUSED" else 0),
            "positive_rate": round(float(mis.mean()), 4),
            "hidden_crisis": int((dirn == 1).sum()),
            "false_alarm": int((dirn == -1).sum()),
            "agreement_with_fused": round(float((mis == fused_mis).mean()), 4),
        })
    return pd.DataFrame(rows)

def pairwise_agreement(df, tau) -> dict:
    dirs = {}
    for name, col in [("text", "sev_text"), ("embed", "sev_embed"), ("rt", "sev_rt")]:
        delta = df[col].values - df["priority_score"].values
        dirs[name] = direction(delta, tau)
    out = {}
    for a, b in [("text", "embed"), ("text", "rt"), ("embed", "rt")]:
        out[f"{a}_vs_{b}"] = round(float((dirs[a] == dirs[b]).mean()), 4)
    return out

# --------------------------------------------------------------------------- #
def main():
    df = pd.read_parquet(C.PROC_DIR / "processed.parquet")
    print(f"[load] {len(df):,} rows")

    print("[signals] computing rule + embedding + resolution-time ...")
    df = S.compute_all_signals(df)

    df["inferred_severity"] = fuse(df)
    df["inferred_level"] = df["inferred_severity"].map(C.score_to_priority)
    df["severity_delta"] = df["inferred_severity"] - df["priority_score"]
    df["level_delta"] = (df["inferred_level"].map(C.PRIORITY_ORD)
                         - df[C.COL_PRIORITY].map(C.PRIORITY_ORD))

    # calibrate TAU on TRAIN only (no peeking at the held-out split)
    tr = df["split"] == "train"
    tau = calibrate_tau(df.loc[tr, "severity_delta"].values)

    dirn = direction(df["severity_delta"].values, tau)
    df["mismatch"] = (dirn != 0).astype(int)
    df["mismatch_type"] = np.where(dirn == 1, C.HIDDEN_CRISIS,
                            np.where(dirn == -1, C.FALSE_ALARM, "—"))

    # ----- metrics -----
    pos_overall = float(df["mismatch"].mean())
    pos_train = float(df.loc[tr, "mismatch"].mean())
    pos_test = float(df.loc[~tr, "mismatch"].mean())
    agree = pairwise_agreement(df, tau)
    ablation = build_ablation(df, tau)

    stats = {
        "tau": round(tau, 4),
        "positive_rate_overall": round(pos_overall, 4),
        "positive_rate_train": round(pos_train, 4),
        "positive_rate_test": round(pos_test, 4),
        "n_hidden_crisis": int((dirn == 1).sum()),
        "n_false_alarm": int((dirn == -1).sum()),
        "signal_agreement": agree,
        "primary_signal_agreement_text_vs_embed": agree["text_vs_embed"],
        "fusion_weights": C.SIGNAL_WEIGHTS,
    }

    df.to_parquet(C.PROC_DIR / "pseudo_labeled.parquet", index=False)
    # PII-free slim file for the public dashboard (drops Customer_Name / Customer_Email)
    dash_cols = [C.COL_PRIORITY, C.COL_CATEGORY, C.COL_CHANNEL, C.COL_AGENT, C.COL_SAT,
                 C.COL_RES_HRS, "domain_tier", "sev_text", "sev_embed", "sev_rt",
                 "inferred_severity", "inferred_level", "severity_delta", "level_delta",
                 "mismatch", "mismatch_type", "urgency_cluster", "split"]
    df[[c for c in dash_cols if c in df.columns]].to_parquet(
        C.PROC_DIR / "dashboard.parquet", index=False)
    with open(C.METRICS_DIR / "pseudolabel_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    ablation.to_csv(C.METRICS_DIR / "ablation.csv", index=False)

    # ----- report -----
    print(f"\n[TAU] {tau:.3f}  -> positive rate: overall={pos_overall:.3f} "
          f"train={pos_train:.3f} test={pos_test:.3f}")
    print(f"[types] Hidden Crisis={stats['n_hidden_crisis']:,}  "
          f"False Alarm={stats['n_false_alarm']:,}")
    print(f"[signal agreement] {agree}")
    print("\n[ABLATION — each signal's individual contribution]")
    print(ablation.to_string(index=False))

    print("\n[sample HIDDEN CRISIS (under-prioritized)]")
    for _, r in df[df.mismatch_type == C.HIDDEN_CRISIS].head(4).iterrows():
        print(f"  [{r[C.COL_PRIORITY]}->{r['inferred_level']}] d={r['severity_delta']:+.2f} "
              f"| {r['lead_sentence'][:70]}")
    print("[sample FALSE ALARM (over-prioritized)]")
    for _, r in df[df.mismatch_type == C.FALSE_ALARM].head(4).iterrows():
        print(f"  [{r[C.COL_PRIORITY]}->{r['inferred_level']}] d={r['severity_delta']:+.2f} "
              f"| {r['lead_sentence'][:70]}")
    print(f"\n[saved] pseudo_labeled.parquet + pseudolabel_stats.json + ablation.csv")
    return df

if __name__ == "__main__":
    main()
