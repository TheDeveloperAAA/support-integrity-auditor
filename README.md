# 🛡️ Support Integrity Auditor (SIA)

A **semantics-driven, evidence-grounded** auditor that detects **Priority Mismatch** in
CRM support tickets — cases where a ticket's objective severity conflicts with its
human-assigned priority — and emits a **hallucination-free Evidence Dossier** for every
flagged case.

There are **no pre-annotated mismatch labels**. SIA bootstraps its own supervision
signal from raw ticket data (self-supervised), fine-tunes a classifier on those
pseudo-labels, and generalizes to unseen and adversarial tickets.

| Held-out verification gate | Required | Achieved |
|---|---|---|
| Binary accuracy | ≥ 83% | **91.50%**  |
| Macro F1 | ≥ 0.82 | **0.874** |
| Per-class recall (both) | ≥ 0.78 | **0.909** / **0.944** |
| Adversarial robustness (bonus) | ≥ 7/10 | **8/10**  (+10%) |
| Dossier hallucinations | 0 | **0** |

---

## 1. Approach

```
Raw tickets (NO mismatch labels)
        │
   STAGE 1 ── fuse 3 independent severity signals ──► INFERRED severity (0–1)
   (self-       • rule-based lexical   (label-independent, interpretable)
    supervised) • neural semantic      (sentence-transformers anchors; adversary-robust)
        │       • resolution-time      (corroborator only — it is an EFFECT of the label)
        │
        ├─ severity_delta = inferred_severity − assigned_priority_score
        │    |delta| ≥ TAU  →  mismatch     (TAU auto-calibrated on TRAIN only)
        │    delta > 0 → Hidden Crisis (under-prioritized)
        │    delta < 0 → False Alarm   (over-prioritized)
        │
   STAGE 2 ── fine-tune distilroberta-base on (assigned priority + channel/category/
   (supervised)  tier tags + real issue sentence) → binary Consistent/Mismatched,
        │        class-weighted loss for imbalance.  Held-out verification.
        │
   STAGE 3 ── Evidence Dossier per flagged ticket (exact schema, every evidence item
   (explain)     copied from a real field; validator rejects anything untraceable).
```

**Why resolution time is only a corroborator.** RT is *caused by* the assigned
priority: a hidden crisis marked `Low` gets deprioritized → resolves slowly → *looks*
low-severity; a false alarm marked `High` resolves fast → *looks* severe. So RT tends to
**confirm the label and mask the very mismatches we hunt**. The label-independent text
signals therefore drive inference; RT is weighted lowest (0.20). The ablation below shows
RT alone over-flags with the lowest agreement.

**Adversarial robustness by design.** Pure-urgency adjectives (`urgent`, `asap`,
`emergency`, `critical`) carry a deliberately tiny weight (0.04) while *substantive* terms
(`crash`, `charged twice`, `data corruption`, `cannot process`) stay strong. Keyword
stuffing therefore cannot inflate a trivial request — directly defeating the
keyword-anchoring attack the project targets.

## 2. Dataset

`customer_support_tickets.csv` — 20,000 rows, 12 columns, zero nulls, no duplicate IDs.

> **Not included in this repo** (it contains 20,000 customer names/emails — PII).
> Download from
> [kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset](https://www.kaggle.com/datasets/ajverse/customer-support-tickets-crm-dataset)
> and place it at `data/customer_support_tickets.csv`, then run `python3 train_pipeline.py`.

> **Schema note:** the actual columns differ from the problem-statement PDF. We build
> against the real names: `Priority_Level` (label audited), `Issue_Category`,
> `Resolution_Time_Hours`, `Ticket_Channel` (only Chat/Email/Web Form), `Customer_Email`
> (tier proxy — ~30% business domains: enterprise.org/company.com/tech.io). Bonus columns
> `Satisfaction_Score`, `Assigned_Agent`, `Submission_Date` are not in the PDF.

Key profiling insights that shaped the design:
- **Category → Priority is near-deterministic** (Fraud only High/Critical; Critical only
  in Technical/Fraud) → category is a strong severity prior.
- **Resolution time is inversely coupled to priority** (Critical ≈12 h → Low ≈45 h) and
  likely generated *from* the label → corroborator only.
- Descriptions = one genuine issue sentence + faker filler → we extract the **leading
  sentence**; raw keyword counts over the filler are unreliable.

## 3. Stage-1 ablation (each signal's individual contribution)

Self-supervised mismatch derivation, TAU = 0.395, fused positive rate **18.3%**.
Pseudo-Label Signal Agreement (text vs embedding) = **0.657**.

| Signal | Weight | Positive rate | Hidden Crisis | False Alarm | Agreement w/ fused |
|---|---|---|---|---|---|
| rule-based lexical | 0.45 | 26.5% | 2,383 | 2,923 | 0.797 |
| neural semantic | 0.35 | 34.9% | 6,056 | 916 | 0.775 |
| resolution-time | 0.20 | 33.7% | 6,334 | 409 | 0.713 |
| **FUSED** | — | **18.3%** | **3,133** | **525** | **1.000** |

Each signal alone over-flags in a different direction (lexical inflates false alarms,
embedding/RT inflate hidden crises); the weighted fusion is the disciplined estimator,
and RT has the lowest agreement — empirical confirmation it tracks the label, not truth.

## 4. Stage-2 classifier

- **Model:** fine-tuned `distilroberta-base` (the spec's DeBERTa-v3-small example crashes
  on Apple MPS via a float16/float32 op bug; on CUDA, swap `BASE_MODEL` back — see
  `src/config.py`). This is a *fine-tuned*, not frozen, model as required.
- **Inputs:** `model_text` = assigned priority + channel/category/tier tags + the real
  issue sentence → text **plus** structured metadata features.
- **Imbalance:** inverse-frequency class-weighted cross-entropy.
- **Held-out test (3,000 tickets):**

```
Accuracy          : 91.50%   (>= 83%)   PASS
Macro F1          : 0.8743   (>= 0.82)  PASS
Recall Consistent : 0.9086   (>= 0.78)  PASS
Recall Mismatch   : 0.9436   (>= 0.78)  PASS
Confusion matrix  : [[2226, 224], [31, 519]]   (TN FP / FN TP)
```

**Hybrid robustness layer (deployed).** The shipped auditor flags a mismatch if the
classifier OR the calibrated Stage-1 signal fires. The classifier-alone metrics above are
the verification of record; the override is a transparent safety net for out-of-distribution
tickets and lifts the independent adversarial score **5/10 → 8/10** (e.g. `General Inquiry`
or `Account` tickets marked `Critical` — combinations never seen in training).

## 5. Stage-3 Evidence Dossier

Emitted for every flagged ticket, exact schema:

```json
{
  "ticket_id": "...", "assigned_priority": "...", "inferred_severity": "...",
  "mismatch_type": "Hidden Crisis | False Alarm", "severity_delta": "...",
  "feature_evidence": [
    {"signal": "keyword", "value": "...", "weight": "..."},
    {"signal": "resolution_time", "value": "...", "interpretation": "..."}
  ],
  "constraint_analysis": "<2-3 sentence grounded explanation>", "confidence": 0.0
}
```

**Zero-hallucination guarantee:** `feature_evidence` values are only ever copied from real
ticket fields (matched keyword terms, the verbatim issue sentence, the actual resolution
hours, the real category/domain). `src/dossier.py::validate_dossier()` re-checks every
item against the row and rejects anything untraceable.

## 6. Repository layout

```
src/config.py        all design constants (grounded in profiling)
src/data_prep.py     Stage 1 — load, lead-sentence extraction, features, fixed split
src/signals.py       Stage-1 signals (rule / semantic / resolution-time)
src/pseudo_label.py  Stage-2 — fuse, derive mismatch, calibrate TAU, ablation
src/train.py         Stage-3 — fine-tune + held-out verification
src/dossier.py       Evidence Dossier (field-grounded) + validator
src/adversarial.py   10 trick-ticket robustness eval
train_pipeline.py    one-command reproducible pipeline
predict.py           CSV in → predictions + dossiers
app/streamlit_app.py web app (audit · batch · dashboard · heatmap · agent-bias)
notebook.ipynb       full reproducible walkthrough
```

## 7. Reproduce

```bash
pip install -r requirements.txt
python3 train_pipeline.py          # data prep → pseudo-label → fine-tune → verify
python3 src/adversarial.py         # adversarial robustness score
python3 predict.py --input data/adversarial_tickets.csv --out_dir artifacts/predictions
streamlit run app/streamlit_app.py # local app
```

## 8. Live demo & deployment

- **Streamlit app (hosted):** https://huggingface.co/spaces/rajtheman/support-integrity-auditor
- **Source:** https://github.com/TheDeveloperAAA/support-integrity-auditor

Deployment is automated in `deploy.py` (`--github` / `--hf`).

## 9. Notes & limitations

- The dataset is synthetic; email tiers and resolution-time coupling are by construction.
- Pseudo-labels are the evaluation ground truth (no human mismatch labels exist) — metrics
  measure fidelity to the fused self-supervised signal generalized by the classifier.
- `Resolution_Time_Hours` is post-resolution; it is used for pseudo-labeling but excluded
  from the classifier inputs to keep an honest intake-time audit framing.
