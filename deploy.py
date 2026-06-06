"""
Phase 8 — automated deployment.

  --github : create + push the GitHub repo  (TheDeveloperAAA, Contents+Admin token)
  --hf     : create + push the HuggingFace Space (rajtheman, write token) -> live URL

Tokens are read from ~/.sia_gh_token / ~/.sia_hf_token, never written into the repo.
GitHub auth uses git's http.extraheader so the token is not persisted in .git/config.
"""
from __future__ import annotations
import os, sys, json, subprocess, argparse, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import config as C

GH_USER = "TheDeveloperAAA"

def _tok(p: Path) -> str:
    return p.read_text().strip()

# --------------------------------------------------------------------------- #
def deploy_github():
    gh = _tok(C.GH_TOKEN_FILE)
    repo = C.GH_REPO_NAME
    # 1) create repo (idempotent) via curl (system certs; urllib SSL is flaky on macOS)
    body = json.dumps({"name": repo, "private": False,
                       "description": "Support Integrity Auditor (SIA) — self-supervised priority-mismatch auditor"})
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
                        "-X", "POST", "-H", f"Authorization: Bearer {gh}",
                        "-H", "Accept: application/vnd.github+json",
                        "https://api.github.com/user/repos", "-d", body],
                       capture_output=True, text=True)
    print(f"[github] create repo -> HTTP {r.stdout}  (201=created, 422=already exists)")

    url = f"https://github.com/{GH_USER}/{repo}.git"
    hdr = f"http.extraheader=AUTHORIZATION: bearer {gh}"
    def git(*args, check=True):
        return subprocess.run(["git", "-c", "user.email=sia@local",
                               "-c", "user.name=SIA Bot", *args],
                              cwd=ROOT, check=check, capture_output=True, text=True)
    git("add", "-A")
    git("commit", "-m", "Support Integrity Auditor — full pipeline, app, artifacts", check=False)
    git("branch", "-M", "main")
    push = subprocess.run(["git", "-c", hdr, "push", "--force", url, "main"],
                          cwd=ROOT, capture_output=True, text=True)
    if push.returncode == 0:
        print(f"[github] pushed -> https://github.com/{GH_USER}/{repo}")
    else:
        print("[github] push failed:\n", push.stderr[-800:])

# --------------------------------------------------------------------------- #
SPACE_README = """---
title: Support Integrity Auditor
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: streamlit
sdk_version: {sdk}
app_file: app/streamlit_app.py
pinned: false
license: mit
---

# Support Integrity Auditor (SIA)

Self-supervised auditor that detects **Priority Mismatch** in CRM support tickets —
tickets whose human-assigned priority conflicts with their true severity — and emits
a hallucination-free **Evidence Dossier** for every flagged case.

- **Audit a Ticket** — single-ticket form → binary judgment + dossier
- **Batch Audit** — CSV upload → predictions + downloadable dossiers
- **Mismatch Dashboard** — flagged distribution, mismatch types, severity-delta heatmap, agent-bias view
"""

def deploy_hf():
    import streamlit
    from huggingface_hub import HfApi
    hf = _tok(C.HF_TOKEN_FILE)
    api = HfApi(token=hf)
    user = api.whoami()["name"]
    repo_id = f"{user}/{C.HF_SPACE_NAME}"
    api.create_repo(repo_id, repo_type="space", space_sdk="streamlit", exist_ok=True)

    # Space README with front-matter (separate from the repo's methodology README)
    sp = ROOT / ".hf_space_README.md"
    sp.write_text(SPACE_README.format(sdk=streamlit.__version__))
    api.upload_file(path_or_fileobj=str(sp), path_in_repo="README.md",
                    repo_id=repo_id, repo_type="space")
    sp.unlink()

    api.upload_folder(
        folder_path=str(ROOT), repo_id=repo_id, repo_type="space",
        ignore_patterns=[
            ".git*", ".claude/**", "**/__pycache__/**", "*.pyc", ".sia_*", "*_token",
            "*.hf_token", "*.gh_token", "README.md", ".hf_space_README.md",
            # raw PII dataset (names/emails) — never publish
            "data/SIA.pdf", "data/customer_support_tickets.csv",
            "data/enhanced_customer_support_data.csv",
            # PII-bearing intermediate parquets — keep only the slim dashboard.parquet
            "artifacts/data/pseudo_labeled.parquet", "artifacts/data/processed.parquet",
            "artifacts/data/test_predictions.parquet", "artifacts/predictions/**",
        ],
    )
    print(f"[hf] Space live -> https://huggingface.co/spaces/{repo_id}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--github", action="store_true")
    ap.add_argument("--hf", action="store_true")
    a = ap.parse_args()
    if not (a.github or a.hf):
        a.github = a.hf = True
    if a.github: deploy_github()
    if a.hf: deploy_hf()
