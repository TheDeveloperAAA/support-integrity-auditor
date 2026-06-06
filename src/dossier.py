"""
Phase 4 — Evidence Dossier generation (hallucination-free by construction).

For every ticket flagged as a mismatch we emit the EXACT schema from the problem
statement.  Design guarantee: feature_evidence values are only ever copied from
real ticket fields (matched keyword terms, the verbatim issue sentence, the actual
resolution hours, the real category/domain).  Nothing is invented.  validate_dossier()
re-checks every item against the row and rejects anything untraceable.

This module is lightweight (no model load); predict.py computes the signals first
then calls build_dossier per flagged row.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import signals as S

# typical resolution hours per assigned priority (from profiling) — used only to
# describe the actual hours relative to a known baseline (grounded comparison).
_TYPICAL_RT = {"Low": 45.0, "Medium": 44.5, "High": 24.5, "Critical": 12.0}
_KW_WEIGHT = {"critical": "0.26 (critical cue)", "high": "0.11 (high-severity cue)",
              "urgency": "0.04 (urgency phrasing — discounted)",
              "trivial": "-0.22 (trivial/routine cue, lowers severity)"}


def _resolution_interpretation(hours: float, assigned: str) -> str:
    exp = _TYPICAL_RT.get(assigned, 39.0)
    if hours > exp * 1.3:
        return f"{hours:.0f}h to resolve — slower than typical for '{assigned}' (~{exp:.0f}h)"
    if hours < exp * 0.7:
        return f"{hours:.0f}h to resolve — faster than typical for '{assigned}' (~{exp:.0f}h)"
    return f"{hours:.0f}h to resolve — near typical for '{assigned}' (~{exp:.0f}h)"


def build_dossier(row, prob: float | None = None) -> dict:
    """Construct the exact-schema dossier for one flagged ticket."""
    assigned = str(row[C.COL_PRIORITY])
    inferred = str(row["inferred_level"])
    subject = str(row[C.COL_SUBJECT])
    lead = str(row["lead_sentence"])
    delta = float(row["severity_delta"])
    level_delta = int(row["level_delta"])
    mtype = str(row["mismatch_type"])
    hours = float(row[C.COL_RES_HRS])
    category = str(row[C.COL_CATEGORY])
    tier = str(row.get("domain_tier", C.domain_tier(row[C.COL_EMAIL])))

    evidence = []

    # (1) verbatim issue sentence — always present, always traceable to the description
    evidence.append({
        "signal": "text",
        "value": lead,
        "interpretation": f"semantic content reads as '{inferred}' severity",
    })

    # (2) matched keyword terms (only real matches from subject + description)
    ev = S.text_rule_evidence(f"{subject}. {lead}")
    for bucket in ("critical", "high", "urgency", "trivial"):
        for term in ev[bucket]:
            evidence.append({
                "signal": "keyword",
                "value": term,
                "weight": _KW_WEIGHT[bucket],
            })

    # (3) resolution time — the actual hours field, with a grounded comparison
    evidence.append({
        "signal": "resolution_time",
        "value": f"{hours:.0f}h",
        "interpretation": _resolution_interpretation(hours, assigned),
    })

    # (4) category prior — only when the category is itself a strong severity signal
    if category in ("Fraud", "Technical") and mtype == C.HIDDEN_CRISIS:
        evidence.append({
            "signal": "category",
            "value": category,
            "interpretation": f"'{category}' issues skew high-severity in this dataset",
        })

    # (5) business-tier customer raises the stakes of an under-prioritized ticket
    if tier == "business" and mtype == C.HIDDEN_CRISIS:
        evidence.append({
            "signal": "customer_tier",
            "value": str(row[C.COL_EMAIL]).split("@")[-1],
            "interpretation": "business-domain customer — higher impact if under-served",
        })

    # constraint analysis — built only from the values above
    if mtype == C.HIDDEN_CRISIS:
        verb = "under-prioritized"
        why = (f"The content indicates '{inferred}' severity, "
               f"yet it was logged as '{assigned}'.")
    else:
        verb = "over-prioritized"
        why = (f"The request is routine ('{inferred}'-level), "
               f"yet it was escalated to '{assigned}'.")
    constraint_analysis = (
        f"{why} Resolution took {hours:.0f}h ({_resolution_interpretation(hours, assigned).split('—')[1].strip()}). "
        f"This {verb} ticket is a {mtype} (severity gap {level_delta:+d} levels)."
    )

    if prob is None:
        prob = min(0.99, 0.55 + abs(delta))
    return {
        "ticket_id": str(row[C.COL_ID]),
        "assigned_priority": assigned,
        "inferred_severity": inferred,
        "mismatch_type": mtype,
        "severity_delta": f"{level_delta:+d} levels ({assigned}->{inferred}); score Δ={delta:+.2f}",
        "feature_evidence": evidence,
        "constraint_analysis": constraint_analysis,
        "confidence": round(float(prob), 3),
    }


def validate_dossier(dossier: dict, row) -> list[str]:
    """Return a list of hallucination violations (empty == clean).

    Every feature_evidence value must be verifiable against the actual ticket row.
    """
    problems = []
    hay = f"{row[C.COL_SUBJECT]}. {row['lead_sentence']}".lower()
    real_hours = f"{float(row[C.COL_RES_HRS]):.0f}h"
    real_domain = str(row[C.COL_EMAIL]).split("@")[-1].lower()
    for item in dossier["feature_evidence"]:
        sig, val = item["signal"], str(item["value"])
        if sig in ("text",):
            if val.lower() not in hay and hay not in val.lower():
                problems.append(f"text not in ticket: {val!r}")
        elif sig == "keyword":
            if val.lower() not in hay:
                problems.append(f"keyword not in ticket: {val!r}")
        elif sig == "resolution_time":
            if val != real_hours:
                problems.append(f"resolution_time mismatch: {val} != {real_hours}")
        elif sig == "category":
            if val != str(row[C.COL_CATEGORY]):
                problems.append(f"category mismatch: {val}")
        elif sig == "customer_tier":
            if val.lower() != real_domain:
                problems.append(f"domain mismatch: {val}")
    return problems
