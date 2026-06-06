"""
Central configuration for the Support Integrity Auditor (SIA).

Every constant here is grounded in the dataset profiling pass (see README §Data).
Keeping all design decisions in one place makes the pipeline reproducible and the
ablation/threshold choices auditable.
"""
from __future__ import annotations
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
PROC_DIR = ARTIFACTS / "data"
METRICS_DIR = ARTIFACTS / "metrics"

RAW_CSV = DATA_DIR / "customer_support_tickets.csv"
CLASSIFIER_DIR = MODELS_DIR / "deberta-sia"      # fine-tuned classifier output

for _d in (ARTIFACTS, MODELS_DIR, PROC_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# --------------------------------------------------------------------------- #
# Schema — ACTUAL column names in the CSV (NOT the PDF's names).
# --------------------------------------------------------------------------- #
COL_ID       = "Ticket_ID"
COL_NAME     = "Customer_Name"
COL_EMAIL    = "Customer_Email"
COL_SUBJECT  = "Ticket_Subject"
COL_DESC     = "Ticket_Description"
COL_CATEGORY = "Issue_Category"
COL_PRIORITY = "Priority_Level"          # the human label being audited
COL_CHANNEL  = "Ticket_Channel"
COL_DATE     = "Submission_Date"
COL_RES_HRS  = "Resolution_Time_Hours"
COL_AGENT    = "Assigned_Agent"
COL_SAT      = "Satisfaction_Score"

TEXT_COLS = [COL_SUBJECT, COL_DESC]

# --------------------------------------------------------------------------- #
# Priority on a common 0..1 severity axis.
# Anchors chosen to spread the 4 ordinal levels across [0,1].
# --------------------------------------------------------------------------- #
PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]
PRIORITY_ORD = {lvl: i for i, lvl in enumerate(PRIORITY_LEVELS)}
PRIORITY_TO_SCORE = {"Low": 0.12, "Medium": 0.40, "High": 0.70, "Critical": 0.92}

# Map a continuous inferred-severity score back to an ordinal label.
def score_to_priority(score: float) -> str:
    if score < 0.30:  return "Low"
    if score < 0.55:  return "Medium"
    if score < 0.80:  return "High"
    return "Critical"

# --------------------------------------------------------------------------- #
# Category severity prior — from the Priority x Category crosstab.
# Fraud is only ever High/Critical; Technical spans all; General Inquiry skews low.
# --------------------------------------------------------------------------- #
CATEGORY_SEVERITY_PRIOR = {
    "Fraud":           0.90,
    "Technical":       0.58,
    "Billing":         0.45,
    "Account":         0.40,
    "General Inquiry": 0.25,
}

# --------------------------------------------------------------------------- #
# Customer tier proxy via email domain (~30% of customers are business-tier).
# --------------------------------------------------------------------------- #
BUSINESS_DOMAINS = {"enterprise.org", "company.com", "tech.io"}
CONSUMER_DOMAINS = {"example.com", "example.org", "example.net"}
def domain_tier(email: str) -> str:
    dom = email.split("@")[-1].strip().lower() if "@" in str(email) else ""
    return "business" if dom in BUSINESS_DOMAINS else "consumer"

# --------------------------------------------------------------------------- #
# Channels (only 3 exist in the data, balanced).
# --------------------------------------------------------------------------- #
CHANNELS = ["Chat", "Email", "Web Form"]

# --------------------------------------------------------------------------- #
# Resolution time -> severity proxy.
# Observed: min 1, p25 11, median 27, p75 58, p95 120, cap 120h.
# Short resolution time => high severity (SLA: severe tickets resolved fast).
# Empirical anchors used for an inverse mapping (overridden by data quantiles
# computed at runtime in signals.py).
# --------------------------------------------------------------------------- #
RES_CAP_HOURS = 120.0
RES_QUANTILES = {"p25": 11.0, "p50": 27.0, "p75": 58.0, "p95": 120.0}

# --------------------------------------------------------------------------- #
# Rule-based NLP lexicons (Stage-1 text-severity signal).
# Grounded in the actual ticket vocabulary observed during profiling.
# --------------------------------------------------------------------------- #
# SUBSTANTIVE severity terms — describe an actual problem (kept strong).
CRITICAL_TERMS = [
    "fraud", "fraudulent", "unauthorized", "hacked", "breach", "stolen",
    "lawsuit", "legal", "chargeback", "double charged", "charged twice",
    "account locked", "locked out", "data loss", "lost all", "deleted all",
    "corrupt", "corrupting", "corrupted", "data corruption", "security",
    "compromised", "scam", "identity theft",
]
HIGH_TERMS = [
    "crash", "crashes", "crashing", "down", "outage", "not working",
    "cannot", "can't", "won't", "unable", "failed", "failing", "error",
    "broken", "not loading", "spinning wheel", "frozen", "freeze",
    "escalate", "refund", "overcharged", "payment failing", "deadline",
    "cannot process", "lost sales", "lost access", "offline",
]
# URGENCY adjectives — emotional intensity, NOT substance. Deliberately weak so the
# system is robust to keyword-stuffing (the core adversarial attack).
URGENCY_TERMS = [
    "urgent", "asap", "immediately", "emergency", "critical",
    "right now", "right away", "please respond", "please help", "respond asap",
]
TRIVIAL_TERMS = [
    "how do i", "how to", "where is", "what is", "what are", "how does",
    "hours of operation", "business hours", "located", "location", "information",
    "general question", "just wondering", "curious", "upgrade", "discount",
    "pricing", "plan options", "plans", "documentation", "tutorial",
    "password", "reset email", "difference between", "would like to understand",
    "tell me", "let me know", "before i decide", "install", "patch",
]
NEGATIONS = ["not", "no", "never", "cannot", "can't", "won't", "isn't",
             "doesn't", "didn't", "wouldn't", "unable", "fail", "failed"]

# --------------------------------------------------------------------------- #
# Signal fusion (Stage-1). Weights are the DEFAULT; the README ablation reports
# each signal alone and justifies these. text_rule is primary because the genuine
# mismatches live in text-vs-label disagreement; resolution_time is label-correlated
# by construction so it is down-weighted (corroborator, not driver).
# --------------------------------------------------------------------------- #
SIGNAL_WEIGHTS = {
    "text_rule":       0.45,   # lexical severity (label-independent)
    "embedding":       0.35,   # neural semantic severity (label-independent)
    "resolution_time": 0.20,   # corroborator only — RT is an EFFECT of the label
}

# Resolution-time -> severity anchors (hours -> severity), from per-priority mean RT
# observed in profiling (Critical~12h, High~24.5h, Medium/Low~45h). Monotonic decreasing.
RES_SEV_HOURS = [1.0, 12.0, 24.5, 45.0, 120.0]
RES_SEV_VALUES = [1.00, 0.90, 0.70, 0.35, 0.05]

# --------------------------------------------------------------------------- #
# Mismatch derivation.
# mismatch = 1 if |inferred_severity - assigned_priority_score| >= TAU.
# Direction: inferred >> assigned -> Hidden Crisis; inferred << assigned -> False Alarm.
# TAU is calibrated in pseudo_label.py to a target positive rate; this is the seed.
# --------------------------------------------------------------------------- #
MISMATCH_TAU = 0.28
TARGET_POSITIVE_RATE = (0.12, 0.25)      # acceptable mismatch fraction band
HIDDEN_CRISIS = "Hidden Crisis"
FALSE_ALARM = "False Alarm"
CLASS_NAMES = ["Consistent", "Mismatched"]   # 0, 1

# --------------------------------------------------------------------------- #
# Models.
# --------------------------------------------------------------------------- #
# NOTE: DeBERTa-v3-small (the spec's example) crashes on Apple MPS due to an
# internal float16/float32 op mismatch. We use an MPS-stable encoder of similar
# size instead; the requirement is a *fine-tuned* (not frozen) model, which this
# satisfies. On a CUDA box, swap BASE_MODEL back to "microsoft/deberta-v3-small".
BASE_MODEL  = "distilroberta-base"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 128
NUM_LABELS = 2

# Training hyperparameters (tuned for 8GB unified memory / MPS).
TRAIN = {
    "epochs": 3,
    "batch_size": 16,
    "grad_accum": 2,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "test_size": 0.15,        # held-out evaluation split
    "val_size": 0.10,         # of the training remainder
    "use_class_weights": True,
}

# --------------------------------------------------------------------------- #
# Verification thresholds (from the problem statement §6).
# --------------------------------------------------------------------------- #
THRESHOLDS = {
    "accuracy": 0.83,
    "macro_f1": 0.82,
    "per_class_recall": 0.78,
}

# --------------------------------------------------------------------------- #
# Deployment.
# --------------------------------------------------------------------------- #
HF_SPACE_NAME = "support-integrity-auditor"
GH_REPO_NAME = "support-integrity-auditor"
HF_TOKEN_FILE = Path.home() / ".sia_hf_token"
GH_TOKEN_FILE = Path.home() / ".sia_gh_token"
