"""
Phase 2a — Stage-1 severity signals (self-supervised).

Three INDEPENDENT severity estimators, each on a 0..1 scale aligned with the
priority anchors in config.PRIORITY_TO_SCORE:

  1. sev_text   — rule-based NLP (lexical): critical/high/trivial term densities
                  + negation.  Label-independent.  Fully interpretable (its matched
                  terms feed the Evidence Dossier).
  2. sev_embed  — neural semantic urgency via sentence-transformers: cosine
                  similarity of the ticket against SEVERE vs TRIVIAL anchor
                  phrases.  Robust to keyword tricks (adversarial).  Label-independent.
  3. sev_rt     — resolution-time proxy (monotonic inverse).  CORROBORATOR ONLY:
                  RT is an effect of the assigned label, so it is down-weighted.

Also exposes urgency_clusters() (KMeans over embeddings) for semantic urgency
grouping / dashboard, satisfying the 'embedding-based clustering' option literally.
"""
from __future__ import annotations
import os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------- #
# 1. Rule-based lexical severity
# --------------------------------------------------------------------------- #
def _compile(terms):
    # word-ish boundary match, multi-word aware
    return [(t, re.compile(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", re.I)) for t in terms]

_CRIT = _compile(C.CRITICAL_TERMS)
_HIGH = _compile(C.HIGH_TERMS)
_URG = _compile(C.URGENCY_TERMS)
_TRIV = _compile(C.TRIVIAL_TERMS)
_NEG = _compile(C.NEGATIONS)

def text_rule_evidence(text: str) -> dict:
    """Return the exact matched terms per category (used by the dossier)."""
    t = str(text)
    return {
        "critical": [term for term, rx in _CRIT if rx.search(t)],
        "high":     [term for term, rx in _HIGH if rx.search(t)],
        "urgency":  [term for term, rx in _URG if rx.search(t)],
        "trivial":  [term for term, rx in _TRIV if rx.search(t)],
        "negation": [term for term, rx in _NEG if rx.search(t)],
    }

def text_rule_severity(text: str) -> float:
    """Lexical severity in [0,1].  Deterministic and interpretable.

    Substance (critical/high terms) drives severity; pure urgency adjectives get a
    deliberately tiny weight so keyword-stuffing ('urgent urgent urgent') cannot
    inflate a trivial request — the key adversarial-robustness property.
    """
    ev = text_rule_evidence(text)
    nc, nh, nu, nt = (len(ev["critical"]), len(ev["high"]),
                      len(ev["urgency"]), len(ev["trivial"]))
    has_neg = len(ev["negation"]) > 0
    score = 0.42
    score += min(nc, 2) * 0.26          # critical, substantive — dominates
    score += min(nh, 3) * 0.11          # high-severity, substantive
    score += min(nu, 2) * 0.04          # urgency adjectives — weak on purpose
    score -= min(nt, 2) * 0.22          # trivial/routine cues pull down
    if has_neg and nt == 0:
        score += 0.05                   # "cannot / not working" nudges up
    return float(np.clip(score, 0.0, 1.0))

# --------------------------------------------------------------------------- #
# 2. Neural semantic severity (sentence-transformers anchors)
# --------------------------------------------------------------------------- #
SEVERE_ANCHORS = [
    "this is an urgent critical emergency that must be fixed immediately",
    "the system is completely down and not working at all",
    "my account was hacked with unauthorized fraudulent access",
    "I was incorrectly charged twice and need an urgent refund",
    "the application keeps crashing and I cannot use it",
    "nothing is loading, I have lost access to all my data",
    "my account is locked and I am completely blocked from working",
    "this is unacceptable, I will escalate and take legal action",
]
TRIVIAL_ANCHORS = [
    "I have a general question about your hours of operation",
    "where is your headquarters office located",
    "how do I upgrade my subscription plan",
    "just a quick question, nothing urgent",
    "can you point me to the documentation or a tutorial",
    "I would like some general pricing information",
    "what are the features included in the basic plan",
    "I am just curious about how this works",
]

_embedder = None
def load_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _embedder = SentenceTransformer(C.EMBED_MODEL, device=device)
    return _embedder

import json
_EMBED_NORM_FILE = C.METRICS_DIR / "embed_norm.json"

def embed_texts(texts) -> np.ndarray:
    model = load_embedder()
    return model.encode(list(texts), batch_size=128, show_progress_bar=True,
                        normalize_embeddings=True, convert_to_numpy=True)

def _anchor_raw(embeddings: np.ndarray) -> np.ndarray:
    model = load_embedder()
    sev_c = model.encode(SEVERE_ANCHORS, normalize_embeddings=True, convert_to_numpy=True).mean(0)
    triv_c = model.encode(TRIVIAL_ANCHORS, normalize_embeddings=True, convert_to_numpy=True).mean(0)
    sev_c /= (np.linalg.norm(sev_c) + 1e-9)
    triv_c /= (np.linalg.norm(triv_c) + 1e-9)
    return embeddings @ sev_c - embeddings @ triv_c       # cos(severe) - cos(trivial)

def embedding_severity(embeddings: np.ndarray) -> np.ndarray:
    """Map anchor-similarity to [0,1] using DATASET-CALIBRATED absolute constants.

    The (lo, hi) percentiles are computed once on the full corpus and cached, so a
    single-ticket inference gets the same scale as training (no batch dependence).
    """
    raw = _anchor_raw(embeddings)
    if _EMBED_NORM_FILE.exists():
        d = json.load(open(_EMBED_NORM_FILE)); lo, hi = d["lo"], d["hi"]
    else:
        lo, hi = float(np.percentile(raw, 2)), float(np.percentile(raw, 98))
        if len(raw) >= 500:                               # persist from a full run
            json.dump({"lo": lo, "hi": hi}, open(_EMBED_NORM_FILE, "w"))
    if hi - lo < 1e-9:
        return np.full(len(raw), 0.5)
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0)

def urgency_clusters(embeddings: np.ndarray, sev_embed: np.ndarray, k: int = 8):
    """KMeans urgency grouping; each cluster gets a severity = its mean sev_embed.
    Falls back gracefully when there are too few rows (e.g. single-ticket inference)."""
    n = len(embeddings)
    if n < max(k, 2):
        return np.zeros(n, dtype=int), {0: float(np.mean(sev_embed)) if n else 0.5}
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=C.SEED, n_init=10)
    labels = km.fit_predict(embeddings)
    cluster_sev = {c: float(sev_embed[labels == c].mean()) for c in range(k)}
    return labels, cluster_sev

# --------------------------------------------------------------------------- #
# 3. Resolution-time severity (corroborator)
# --------------------------------------------------------------------------- #
def resolution_severity(hours) -> np.ndarray:
    h = np.asarray(hours, dtype=float)
    return np.interp(h, C.RES_SEV_HOURS, C.RES_SEV_VALUES)

# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def compute_all_signals(df):
    """Return df with sev_text, sev_embed, sev_rt, urgency_cluster columns."""
    out = df.copy()
    # signal 1 — lexical (on subject + real issue sentence)
    rule_input = (out[C.COL_SUBJECT].astype(str) + ". " + out["lead_sentence"].astype(str))
    out["sev_text"] = rule_input.map(text_rule_severity).astype(float)
    # signal 2 — neural semantic
    emb = embed_texts(out["lead_sentence"].tolist())
    out["sev_embed"] = embedding_severity(emb)
    labels, cluster_sev = urgency_clusters(emb, out["sev_embed"].values)
    out["urgency_cluster"] = labels
    out["cluster_severity"] = out["urgency_cluster"].map(cluster_sev)
    # signal 3 — resolution time (corroborator)
    out["sev_rt"] = resolution_severity(out[C.COL_RES_HRS].values)
    return out
