"""
Standalone, reproducible training pipeline for the Support Integrity Auditor.

Runs all three mandatory stages end to end:
  STAGE 1  data preparation            (src/data_prep.py)
  STAGE 2  self-supervised pseudo-label generation + ablation (src/pseudo_label.py)
  STAGE 3  fine-tune the binary classifier + held-out verification (src/train.py)

Usage:
  python3 train_pipeline.py            # full pipeline
  python3 train_pipeline.py --skip-train   # regenerate data + pseudo-labels only
"""
from __future__ import annotations
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true",
                    help="run data prep + pseudo-labeling but skip fine-tuning")
    args = ap.parse_args()

    from src import data_prep, pseudo_label
    t0 = time.time()

    print("\n" + "=" * 60 + "\nSTAGE 1/3 — DATA PREPARATION\n" + "=" * 60)
    data_prep.main()

    print("\n" + "=" * 60 + "\nSTAGE 2/3 — PSEUDO-LABEL GENERATION (self-supervised)\n" + "=" * 60)
    pseudo_label.main()

    if not args.skip_train:
        print("\n" + "=" * 60 + "\nSTAGE 3/3 — CLASSIFIER FINE-TUNING + VERIFICATION\n" + "=" * 60)
        from src import train
        train.main()

    print(f"\n[pipeline] complete in {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
