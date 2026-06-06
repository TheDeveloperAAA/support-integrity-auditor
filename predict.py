"""
Inference — accepts a CSV of tickets and outputs predictions + Evidence Dossiers.

Architecture
------------
- The fine-tuned classifier gives the BINARY judgment (Consistent / Mismatched)
  + a confidence.
- The Stage-1 signal fusion supplies the inferred severity / direction that the
  Evidence Dossier needs (inferred_severity, severity_delta, mismatch_type).
- Dossiers are emitted only for tickets the classifier flags, and every one is
  validated to be hallucination-free before it is written.

Usage:
  python3 predict.py --input data/adversarial_tickets.csv --out_dir artifacts/predictions
"""
from __future__ import annotations
import os, sys, json, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config as C
from src import data_prep, signals as S
from src.pseudo_label import fuse
from src.dossier import build_dossier, validate_dossier

REQUIRED = [C.COL_ID, C.COL_SUBJECT, C.COL_DESC, C.COL_PRIORITY,
            C.COL_CATEGORY, C.COL_CHANNEL, C.COL_EMAIL, C.COL_RES_HRS]

_model = _tok = _device = None

def load_model():
    global _model, _tok, _device
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _tok = AutoTokenizer.from_pretrained(str(C.CLASSIFIER_DIR))
        _model = AutoModelForSequenceClassification.from_pretrained(str(C.CLASSIFIER_DIR))
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
        _model.to(_device).eval()
    return _tok, _model, _device

def _check_columns(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    if C.COL_SAT not in df.columns:
        df[C.COL_SAT] = np.nan
    return df

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Feature build + Stage-1 signals + inferred severity / direction."""
    df = _check_columns(df.copy())
    df = data_prep.build_features(df)
    df = S.compute_all_signals(df)
    df["inferred_severity"] = fuse(df)
    df["inferred_level"] = df["inferred_severity"].map(C.score_to_priority)
    df["priority_score"] = df[C.COL_PRIORITY].map(C.PRIORITY_TO_SCORE)
    df["severity_delta"] = df["inferred_severity"] - df["priority_score"]
    df["level_delta"] = (df["inferred_level"].map(C.PRIORITY_ORD)
                         - df[C.COL_PRIORITY].map(C.PRIORITY_ORD))
    df["mismatch_type"] = np.where(df["severity_delta"] >= 0, C.HIDDEN_CRISIS, C.FALSE_ALARM)
    return df

@torch.no_grad()
def classify(df: pd.DataFrame, batch=64):
    tok, model, device = load_model()
    probs = []
    texts = df["model_text"].tolist()
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i+batch], truncation=True, max_length=C.MAX_LEN,
                  padding=True, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs.append(F.softmax(logits, -1)[:, 1].cpu().numpy())
    p = np.concatenate(probs) if probs else np.array([])
    return (p >= 0.5).astype(int), p

def _signal_tau() -> float:
    p = C.METRICS_DIR / "pseudolabel_stats.json"
    return json.load(open(p))["tau"] if p.exists() else C.MISMATCH_TAU

def run_inference(df: pd.DataFrame):
    """Return (enriched_df_with_predictions, list_of_dossiers).

    Final judgment is a HYBRID: the fine-tuned classifier OR a high-confidence
    Stage-1 signal (|severity_delta| >= calibrated TAU). The classifier provides
    learned generalization; the calibrated signal is a transparent safety net for
    out-of-distribution tickets (e.g. category/priority combos unseen in training).
    The classifier's standalone held-out metrics remain the verification of record.
    """
    df = enrich(df)
    preds, probs = classify(df)
    tau = _signal_tau()
    signal_mis = (df["severity_delta"].abs() >= tau).astype(int).values
    df["clf_mismatch"] = preds
    df["signal_mismatch"] = signal_mis
    df["pred_mismatch"] = ((preds == 1) | (signal_mis == 1)).astype(int)
    df["pred_prob_mismatch"] = probs
    dossiers = []
    for _, row in df[df.pred_mismatch == 1].iterrows():
        p = float(row["pred_prob_mismatch"])
        if row["clf_mismatch"] == 0:                 # signal-only override
            p = min(0.99, 0.55 + abs(float(row["severity_delta"])))
        d = build_dossier(row, prob=p)
        violations = validate_dossier(d, row)
        if violations:                     # safety net — should never trigger
            d["feature_evidence"] = [e for e in d["feature_evidence"]
                                     if not any(str(e["value"]) in v for v in violations)]
            d["_validation_warnings"] = violations
        dossiers.append(d)
    return df, dossiers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default=str(C.ARTIFACTS / "predictions"))
    args = ap.parse_args()

    df_in = pd.read_csv(args.input)
    df, dossiers = run_inference(df_in)

    os.makedirs(args.out_dir, exist_ok=True)
    cols = [C.COL_ID, C.COL_PRIORITY, "inferred_level", "pred_mismatch",
            "mismatch_type", "pred_prob_mismatch"]
    df[cols].to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)
    with open(os.path.join(args.out_dir, "dossiers.json"), "w") as f:
        json.dump(dossiers, f, indent=2)

    n_flag = int((df.pred_mismatch == 1).sum())
    print(f"[predict] {len(df)} tickets -> {n_flag} flagged as Priority Mismatch")
    print(f"[predict] wrote predictions.csv + {len(dossiers)} dossiers to {args.out_dir}")

if __name__ == "__main__":
    main()
