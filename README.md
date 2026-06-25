# Redrob Hackathon — Candidate Ranker

## Overview

A two-stage CPU-only candidate ranking system for the Redrob Intelligent Candidate Discovery & Ranking Challenge.

**Architecture:**
1. **JD-fit score** — TF-IDF text similarity + explicit skill matching (required/bonus) + role/career quality + location fit
2. **Behavioral availability multiplier** — last active date, open-to-work flag, recruiter response rate, notice period, interview completion — applied as a multiplier so a ghosted candidate cannot rank in top 10 regardless of skill fit
3. **Honeypot detection** — impossible timeline checks, expert-with-0-months, absurd skill counts

## Key design decisions

- No GPU, no API calls — pure scikit-learn TF-IDF, runs on CPU in < 5 minutes
- Behavioral score is a **multiplier** (not additive): a candidate inactive for 6 months gets base × 0.12, preventing them from surfacing in top 10
- Explicit disqualifiers: pure consulting-firm backgrounds (TCS/Infosys/Wipro/etc.) are penalized; pure research roles penalized; unrelated titles (HR Manager, Accountant) are penalized
- Honeypots caught by: timeline impossibility, expert-proficiency with 0 months, impossible YoE vs career start date

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Full run (produces submission CSV)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Quick test on a small sample
python rank.py --candidates ./sample_candidates.json --out ./test_submission.csv
```

> **Note:** The ranking step (TF-IDF + scoring) runs in well under 5 minutes on a 16GB CPU machine. Pre-computation is not required — all features are computed from the raw JSONL.

## Score weights

| Component | Weight | Notes |
|---|---|---|
| TF-IDF JD similarity | 15% | Broad semantic match |
| Required skill match | 30% | Embeddings, vector DBs, evaluation frameworks, Python |
| Bonus skill match | 8% | LoRA, LTR, HR-tech, open source |
| Role/career fit | 22% | Product vs consulting, title relevance, YoE |
| Location | 8% | Pune/Noida preferred; India acceptable |
| Career quality | 17% | GitHub, assessments, endorsements, education tier |
| × Behavioral multiplier | — | Availability, recency, response rate |

## File structure

```
rank.py              — main ranker
requirements.txt     — dependencies
README.md            — this file
submission_metadata.yaml — metadata (fill in your team details)
```
