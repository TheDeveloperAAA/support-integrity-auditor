"""
Phase 5 — Adversarial robustness evaluation.

Scores the system on 10 held-out tickets engineered to fool keyword-based systems:
urgent-word-stuffed trivia (False Alarms), calm-but-serious crises and sentiment/
negation traps (Hidden Crises), plus 2 genuinely-consistent controls. Each ticket's
resolution time is set to corroborate the WRONG label, so only real semantic
understanding succeeds.

A ticket is 'correct' iff:
  - expected Consistent -> classifier predicts NOT mismatch, OR
  - expected mismatch    -> classifier predicts mismatch AND the direction
                            (Hidden Crisis / False Alarm) matches.
>= 7/10 earns the spec's 10% bonus. Also asserts every dossier is hallucination-free.

Run:  python3 src/adversarial.py
"""
from __future__ import annotations
import os, sys, json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.dossier import validate_dossier

# ground truth: ticket_id -> (is_mismatch, expected_type or None)
EXPECTED = {
    "ADV-001": (True,  C.FALSE_ALARM),    # trivial hours Q stuffed with "urgent/critical", Critical
    "ADV-002": (True,  C.FALSE_ALARM),    # student-discount Q dressed as "emergency", High
    "ADV-003": (True,  C.HIDDEN_CRISIS),  # calm, locked out 4 days + client demo, Low
    "ADV-004": (True,  C.HIDDEN_CRISIS),  # calm, 100x overcharge, Low
    "ADV-005": (True,  C.HIDDEN_CRISIS),  # calm, silent data corruption, Medium
    "ADV-006": (True,  C.HIDDEN_CRISIS),  # positive-word trap, data loss, Low
    "ADV-007": (True,  C.HIDDEN_CRISIS),  # negation trap, payments down, Low
    "ADV-008": (False, None),             # genuine trivial plan Q, Low -> consistent
    "ADV-009": (False, None),             # genuine outage, Critical -> consistent
    "ADV-010": (True,  C.FALSE_ALARM),    # keyword-stuffed password reset, Critical
}

def main():
    from predict import run_inference   # imported here so model loads lazily
    df_in = pd.read_csv(C.DATA_DIR / "adversarial_tickets.csv")
    df, dossiers = run_inference(df_in)
    df = df.set_index(C.COL_ID)
    dossier_by_id = {d["ticket_id"]: d for d in dossiers}

    correct, halluc = 0, 0
    rows = []
    for tid, (exp_mis, exp_type) in EXPECTED.items():
        r = df.loc[tid]
        pred_mis = bool(r["pred_mismatch"])
        pred_type = r["mismatch_type"] if pred_mis else "—"
        if exp_mis:
            ok = pred_mis and (pred_type == exp_type)
        else:
            ok = not pred_mis
        correct += int(ok)
        # dossier grounding check
        if tid in dossier_by_id:
            v = validate_dossier(dossier_by_id[tid], r)
            if v: halluc += 1
        rows.append((tid, "MISMATCH" if exp_mis else "consistent",
                     exp_type or "—", "MISMATCH" if pred_mis else "consistent",
                     pred_type, "✅" if ok else "❌"))

    print(f"{'ID':<8}{'EXPECTED':<12}{'exp_type':<15}{'PRED':<12}{'pred_type':<15}OK")
    for t in rows:
        print(f"{t[0]:<8}{t[1]:<12}{t[2]:<15}{t[3]:<12}{t[4]:<15}{t[5]}")
    print("-" * 64)
    print(f"SCORE: {correct}/10   |   dossier hallucinations: {halluc}")
    print(f"BONUS (>=7/10): {'EARNED ✅ (+10%)' if correct >= 7 else 'not earned'}")

    report = {"score": correct, "total": 10, "hallucinations": halluc,
              "bonus_earned": correct >= 7,
              "detail": [dict(zip(["id","expected","exp_type","pred","pred_type","ok"], t))
                         for t in rows]}
    with open(C.METRICS_DIR / "adversarial_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report

if __name__ == "__main__":
    main()
