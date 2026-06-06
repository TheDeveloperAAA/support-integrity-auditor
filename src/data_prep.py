"""
Phase 1 — Data preparation for the Support Integrity Auditor.

Responsibilities
----------------
1. Load the raw CRM ticket CSV.
2. Extract the *real issue sentence* from each description (the dataset pads a
   genuine leading sentence with random faker words — see README §Data).
3. Build the text the classifier sees + structured metadata features.
4. Create a FIXED, stratified train/test split (held-out evaluation set) that is
   independent of any pseudo-label, so labels can never leak into the split.

Run:  python3 src/data_prep.py
Out:  artifacts/data/processed.parquet  (+ a printed summary)
"""
from __future__ import annotations
import os, re, sys
import pandas as pd

# make `from src import config` work when run as a plain script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C

from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Text extraction
# --------------------------------------------------------------------------- #
_GREETING = re.compile(r"^\s*(hi|hello|hey|dear)\b[^,.:;!?]*[,:]?\s*", re.IGNORECASE)
_FIRST_SENT = re.compile(r"(.+?[.?!])(?:\s|$)")

def extract_leading_sentence(desc: str) -> str:
    """Strip the greeting and return the first sentence (the genuine issue).

    The trailing faker words form a second 'sentence'; we deliberately drop them
    because raw keyword counts over the filler are misleading and are exactly the
    surface an adversarial ticket would attack.
    """
    t = str(desc).strip()
    t = _GREETING.sub("", t, count=1)
    m = _FIRST_SENT.search(t)
    lead = (m.group(1) if m else t).strip()
    return lead or t

def build_model_text(row: pd.Series) -> str:
    """Text fed to DeBERTa: assigned priority + structured tags + the real issue.

    Including the *assigned priority* is intentional and correct — the task is to
    judge whether THAT priority disagrees with the ticket's content. The inferred
    severity / mismatch label is never shown to the model.
    """
    tier = C.domain_tier(row[C.COL_EMAIL])
    return (
        f"priority: {row[C.COL_PRIORITY]} | "
        f"channel: {row[C.COL_CHANNEL]} | "
        f"category: {row[C.COL_CATEGORY]} | "
        f"tier: {tier} | "
        f"{str(row[C.COL_SUBJECT]).strip()}. {row['lead_sentence']}"
    )

# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(C.RAW_CSV)
    df.columns = [c.strip() for c in df.columns]
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[C.COL_RES_HRS] = pd.to_numeric(df[C.COL_RES_HRS], errors="coerce")
    df[C.COL_SAT] = pd.to_numeric(df[C.COL_SAT], errors="coerce")

    df["lead_sentence"] = df[C.COL_DESC].map(extract_leading_sentence)
    df["domain_tier"] = df[C.COL_EMAIL].map(C.domain_tier)
    df["priority_score"] = df[C.COL_PRIORITY].map(C.PRIORITY_TO_SCORE)
    df["category_prior"] = df[C.COL_CATEGORY].map(C.CATEGORY_SEVERITY_PRIOR)
    df["model_text"] = df.apply(build_model_text, axis=1)
    return df

def make_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx_train, idx_test = train_test_split(
        df.index,
        test_size=C.TRAIN["test_size"],
        random_state=C.SEED,
        stratify=df[C.COL_PRIORITY],     # stable strata, independent of labels
    )
    df["split"] = "train"
    df.loc[idx_test, "split"] = "test"
    return df

def main() -> pd.DataFrame:
    df = load_raw()
    print(f"[load] {len(df):,} rows x {df.shape[1]} cols")
    df = build_features(df)
    df = make_split(df)

    out = C.PROC_DIR / "processed.parquet"
    df.to_parquet(out, index=False)

    # summary
    print(f"[split] train={int((df['split']=='train').sum()):,}  "
          f"test={int((df['split']=='test').sum()):,}")
    print("[priority dist]")
    print(df[C.COL_PRIORITY].value_counts(normalize=True).round(3).to_string())
    print("[domain tier dist]")
    print(df["domain_tier"].value_counts(normalize=True).round(3).to_string())
    print("\n[sample lead-sentence extraction]")
    for _, r in df.head(4).iterrows():
        print(f"  RAW : {r[C.COL_DESC][:90]}")
        print(f"  LEAD: {r['lead_sentence']}")
    print("\n[sample model_text]")
    print("  " + df.iloc[1]["model_text"])
    print(f"\n[saved] {out}")
    return df

if __name__ == "__main__":
    main()
