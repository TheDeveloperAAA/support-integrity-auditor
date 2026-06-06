"""
Phase 3 — Stage-2 classifier (fine-tuned DeBERTa-v3-small).

Trains a binary Consistent/Mismatched classifier on the pseudo-labeled data.
- Inputs: model_text  (assigned priority + channel/category/tier tags + real issue
          sentence) -> satisfies "text + >=1 structured metadata feature".
- Imbalance: class-weighted cross-entropy (inverse frequency).
- Held-out evaluation on the FIXED `split=='test'` rows; reports the three
  verification metrics and PASS/FAIL vs the thresholds in config.

Run:  python3 src/train.py
Out:  artifacts/models/deberta-sia/   (model + tokenizer)
      artifacts/metrics/classifier_metrics.json
      artifacts/data/test_predictions.parquet
"""
from __future__ import annotations
import os, sys, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)

torch.manual_seed(C.SEED); np.random.seed(C.SEED)

# --------------------------------------------------------------------------- #
class TicketDS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item

class WeightedTrainer(Trainer):
    def __init__(self, *a, class_weights=None, **kw):
        super().__init__(*a, **kw)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels,
                               weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    logits = logits[0] if isinstance(logits, tuple) else logits
    labels = eval_pred.label_ids
    preds = logits.argmax(-1)
    rec = recall_score(labels, preds, average=None, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "recall_consistent": float(rec[0]),
        "recall_mismatch": float(rec[1]),
    }

def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(C.BASE_MODEL)
    except Exception as e:
        print("[tok] fast failed, falling back to slow:", e)
        return AutoTokenizer.from_pretrained(C.BASE_MODEL, use_fast=False)

# --------------------------------------------------------------------------- #
def main():
    df = pd.read_parquet(C.PROC_DIR / "pseudo_labeled.parquet")
    train_full = df[df.split == "train"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)

    tr_df, val_df = train_test_split(
        train_full, test_size=C.TRAIN["val_size"], random_state=C.SEED,
        stratify=train_full["mismatch"])

    print(f"[data] train={len(tr_df):,} val={len(val_df):,} test={len(test_df):,}")
    print(f"[label balance] train mismatch rate={tr_df['mismatch'].mean():.3f}")

    tok = load_tokenizer()
    def enc(texts):
        return tok(list(texts), truncation=True, max_length=C.MAX_LEN)
    ds_tr = TicketDS(enc(tr_df["model_text"]), tr_df["mismatch"].values)
    ds_val = TicketDS(enc(val_df["model_text"]), val_df["mismatch"].values)
    ds_te = TicketDS(enc(test_df["model_text"]), test_df["mismatch"].values)

    # inverse-frequency class weights
    counts = np.bincount(tr_df["mismatch"].values, minlength=2)
    cw = torch.tensor(counts.sum() / (2.0 * counts), dtype=torch.float)
    print(f"[class weights] consistent={cw[0]:.3f} mismatch={cw[1]:.3f}")

    model = AutoModelForSequenceClassification.from_pretrained(
        C.BASE_MODEL, num_labels=C.NUM_LABELS,
        id2label={0: C.CLASS_NAMES[0], 1: C.CLASS_NAMES[1]},
        label2id={C.CLASS_NAMES[0]: 0, C.CLASS_NAMES[1]: 1})

    args = TrainingArguments(
        output_dir=str(C.CLASSIFIER_DIR),
        num_train_epochs=C.TRAIN["epochs"],
        per_device_train_batch_size=C.TRAIN["batch_size"],
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=C.TRAIN["grad_accum"],
        learning_rate=C.TRAIN["lr"], weight_decay=C.TRAIN["weight_decay"],
        warmup_ratio=C.TRAIN["warmup_ratio"],
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=1,
        logging_steps=50, report_to=[], seed=C.SEED,
        dataloader_num_workers=0, disable_tqdm=False)

    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_val,
        processing_class=tok, data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics, class_weights=cw)

    trainer.train()

    # ---- final held-out evaluation ----
    pred = trainer.predict(ds_te)
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = logits.argmax(-1)
    m = compute_metrics(pred)
    cm = confusion_matrix(test_df["mismatch"].values, preds).tolist()

    th = C.THRESHOLDS
    passed = (m["accuracy"] >= th["accuracy"] and m["macro_f1"] >= th["macro_f1"]
              and m["recall_consistent"] >= th["per_class_recall"]
              and m["recall_mismatch"] >= th["per_class_recall"])
    result = {**m, "confusion_matrix": cm, "thresholds": th, "PASSED": bool(passed),
              "n_test": int(len(test_df))}

    trainer.save_model(str(C.CLASSIFIER_DIR))
    tok.save_pretrained(str(C.CLASSIFIER_DIR))
    with open(C.METRICS_DIR / "classifier_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    test_df = test_df.copy()
    test_df["pred_mismatch"] = preds
    test_df["pred_prob_mismatch"] = F.softmax(torch.tensor(logits), -1)[:, 1].numpy()
    test_df.to_parquet(C.PROC_DIR / "test_predictions.parquet", index=False)

    print("\n==================  HELD-OUT TEST METRICS  ==================")
    print(f"  Accuracy          : {m['accuracy']*100:.2f}%   (>= {th['accuracy']*100:.0f}%)")
    print(f"  Macro F1          : {m['macro_f1']:.4f}    (>= {th['macro_f1']})")
    print(f"  Recall Consistent : {m['recall_consistent']:.4f}    (>= {th['per_class_recall']})")
    print(f"  Recall Mismatch   : {m['recall_mismatch']:.4f}    (>= {th['per_class_recall']})")
    print(f"  Confusion matrix  : {cm}")
    print(f"  >>> VERIFICATION: {'PASSED ✅' if passed else 'FAILED ❌'}")
    print("============================================================")
    return result

if __name__ == "__main__":
    main()
