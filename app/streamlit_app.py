"""
Support Integrity Auditor — Streamlit web app.

Tabs:
  1. Audit a Ticket   — single-ticket form -> binary judgment + Evidence Dossier
  2. Batch Audit      — CSV upload -> predictions + downloadable dossiers
  3. Mismatch Dashboard — flagged distribution, mismatch types, top signals,
                          severity-delta heatmap (category x channel), agent-bias view
"""
from __future__ import annotations
import os, sys, json
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import config as C

st.set_page_config(page_title="Support Integrity Auditor", page_icon="🛡️", layout="wide")

# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def get_predictor():
    import predict
    predict.load_model()
    return predict

@st.cache_data(show_spinner="Loading dashboard data…")
def load_corpus():
    # prefer the PII-free slim file; fall back to the full labeled corpus locally
    for name in ("dashboard.parquet", "pseudo_labeled.parquet"):
        p = C.PROC_DIR / name
        if p.exists():
            return pd.read_parquet(p)
    return None

def judgment_banner(is_mismatch: bool, conf: float):
    if is_mismatch:
        st.error(f"### ⚠️ PRIORITY MISMATCH  ·  confidence {conf:.0%}")
    else:
        st.success(f"### ✅ CONSISTENT  ·  confidence {1-conf:.0%}")

def render_dossier(d: dict):
    c1, c2, c3 = st.columns(3)
    c1.metric("Assigned priority", d["assigned_priority"])
    c2.metric("Inferred severity", d["inferred_severity"])
    c3.metric("Mismatch type", d["mismatch_type"])
    st.caption(f"Severity delta: {d['severity_delta']}")
    st.markdown("**Constraint analysis**")
    st.info(d["constraint_analysis"])
    st.markdown("**Feature evidence** (every item traceable to a ticket field)")
    st.dataframe(pd.DataFrame(d["feature_evidence"]), use_container_width=True, hide_index=True)
    with st.expander("Raw dossier JSON"):
        st.json(d)

# --------------------------------------------------------------------------- #
st.title("🛡️ Support Integrity Auditor (SIA)")
st.caption("Detects **Priority Mismatch** — tickets whose human-assigned priority "
           "conflicts with their true severity — with a hallucination-free Evidence Dossier.")

tab1, tab2, tab3 = st.tabs(["🔍 Audit a Ticket", "📦 Batch Audit", "📊 Mismatch Dashboard"])

# ----- Tab 1: single ticket ------------------------------------------------- #
with tab1:
    with st.form("single"):
        c1, c2 = st.columns(2)
        subject = c1.text_input("Ticket subject", "Data not syncing - Card")
        category = c2.selectbox("Issue category",
                                ["Technical", "Billing", "Account", "General Inquiry", "Fraud"])
        desc = st.text_area("Ticket description",
                            "Hi Support, The dashboard is not loading any data, just a spinning wheel.")
        c3, c4, c5 = st.columns(3)
        priority = c3.selectbox("Assigned priority", C.PRIORITY_LEVELS)
        channel = c4.selectbox("Channel", C.CHANNELS)
        res_hours = c5.number_input("Resolution time (hours)", 1.0, 120.0, 40.0)
        c6, c7 = st.columns(2)
        email = c6.text_input("Customer email", "user@enterprise.org")
        agent = c7.text_input("Assigned agent", "Anya Sharma")
        submitted = st.form_submit_button("Audit ticket", type="primary")

    if submitted:
        predict = get_predictor()
        row = pd.DataFrame([{
            C.COL_ID: "LIVE-0001", C.COL_NAME: "—", C.COL_EMAIL: email,
            C.COL_SUBJECT: subject, C.COL_DESC: desc, C.COL_CATEGORY: category,
            C.COL_PRIORITY: priority, C.COL_CHANNEL: channel,
            C.COL_DATE: "2026-01-01", C.COL_RES_HRS: res_hours,
            C.COL_AGENT: agent, C.COL_SAT: 3,
        }])
        df, dossiers = predict.run_inference(row)
        r = df.iloc[0]
        is_mis = bool(r["pred_mismatch"])
        judgment_banner(is_mis, float(r["pred_prob_mismatch"]))
        if is_mis and dossiers:
            render_dossier(dossiers[0])
        else:
            st.write(f"Inferred severity level: **{r['inferred_level']}** "
                     f"(assigned **{priority}**) — within tolerance.")

# ----- Tab 2: batch --------------------------------------------------------- #
with tab2:
    st.write("Upload a CSV with the standard ticket columns "
             f"({', '.join([C.COL_SUBJECT, C.COL_DESC, C.COL_PRIORITY, C.COL_CATEGORY, C.COL_CHANNEL, C.COL_RES_HRS])}…).")
    up = st.file_uploader("Tickets CSV", type=["csv"])
    if st.button("Run sample (adversarial set)"):
        up = str(C.DATA_DIR / "adversarial_tickets.csv")
    if up is not None:
        predict = get_predictor()
        df_in = pd.read_csv(up)
        with st.spinner(f"Auditing {len(df_in)} tickets…"):
            df, dossiers = predict.run_inference(df_in)
        n_flag = int((df.pred_mismatch == 1).sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Tickets", len(df))
        k2.metric("Flagged mismatches", n_flag)
        k3.metric("Mismatch rate", f"{n_flag/len(df):.1%}")
        show = df[[C.COL_ID, C.COL_PRIORITY, "inferred_level", "pred_mismatch",
                   "mismatch_type", "pred_prob_mismatch"]].copy()
        show["pred_mismatch"] = show["pred_mismatch"].map({1: "MISMATCH", 0: "consistent"})
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download dossiers (JSON)",
                           json.dumps(dossiers, indent=2),
                           "dossiers.json", "application/json")
        if n_flag:
            st.plotly_chart(px.histogram(df[df.pred_mismatch == 1], x="mismatch_type",
                            color="mismatch_type", title="Flagged tickets by mismatch type"),
                            use_container_width=True)

# ----- Tab 3: dashboard ----------------------------------------------------- #
with tab3:
    corpus = load_corpus()
    if corpus is None:
        st.warning("Run `python3 train_pipeline.py` to generate the labeled corpus first.")
    else:
        flagged = corpus[corpus.mismatch == 1]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total tickets", f"{len(corpus):,}")
        k2.metric("Flagged mismatches", f"{len(flagged):,}")
        k3.metric("Mismatch rate", f"{corpus.mismatch.mean():.1%}")
        k4.metric("Hidden Crisis : False Alarm",
                  f"{(flagged.mismatch_type==C.HIDDEN_CRISIS).sum():,} : "
                  f"{(flagged.mismatch_type==C.FALSE_ALARM).sum():,}")

        c1, c2 = st.columns(2)
        rate_cat = (corpus.groupby(C.COL_CATEGORY)["mismatch"].mean()
                    .sort_values(ascending=False).reset_index())
        c1.plotly_chart(px.bar(rate_cat, x=C.COL_CATEGORY, y="mismatch",
                        title="Mismatch rate by category", labels={"mismatch": "rate"}),
                        use_container_width=True)
        type_chan = flagged.groupby([C.COL_CHANNEL, "mismatch_type"]).size().reset_index(name="n")
        c2.plotly_chart(px.bar(type_chan, x=C.COL_CHANNEL, y="n", color="mismatch_type",
                        title="Mismatch type by channel"), use_container_width=True)

        # severity-delta heatmap: category x channel
        heat = corpus.pivot_table(index=C.COL_CATEGORY, columns=C.COL_CHANNEL,
                                  values="severity_delta", aggfunc="mean")
        st.subheader("Severity-delta heatmap (mean inferred − assigned)")
        st.plotly_chart(px.imshow(heat, text_auto=".2f", color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0, aspect="auto"),
                        use_container_width=True)
        st.caption("Positive (red) = under-prioritized on average (hidden-crisis lean); "
                   "negative (blue) = over-prioritized (false-alarm lean).")

        c3, c4 = st.columns(2)
        # agent bias
        agent_rate = (corpus.groupby(C.COL_AGENT)["mismatch"].mean()
                      .sort_values(ascending=False).reset_index())
        c3.plotly_chart(px.bar(agent_rate, x=C.COL_AGENT, y="mismatch",
                        title="Mismatch rate by agent (bias view)",
                        labels={"mismatch": "rate"}), use_container_width=True)
        # top contributing signals
        sig = (corpus.assign(grp=corpus.mismatch.map({1: "flagged", 0: "consistent"}))
               .groupby("grp")[["sev_text", "sev_embed", "sev_rt"]].mean()
               .T.reset_index().melt(id_vars="index", var_name="grp", value_name="mean"))
        c4.plotly_chart(px.bar(sig, x="index", y="mean", color="grp", barmode="group",
                        title="Mean signal severity: flagged vs consistent",
                        labels={"index": "signal"}), use_container_width=True)
