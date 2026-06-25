"""
Redrob Hackathon — Candidate Ranker
====================================
Strategy: Two-stage scoring
  Stage 1: JD-fit score (skill match + role fit + career quality)
  Stage 2: Behavioral availability multiplier (signals)

Design choices:
  - No GPU, no API calls — pure CPU, runs < 5 min on 16GB machine
  - TF-IDF + cosine similarity for fast text matching (no embedding model needed)
  - Explicit JD signal features (required skills, disqualifiers, location, notice period)
  - Honeypot detection via timeline consistency checks
  - Behavioral availability score as a multiplier (not additive), so a ghost candidate
    can never float to the top even with perfect skills

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Install deps:
    pip install scikit-learn pandas numpy
"""

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# JD CONSTANTS  (derived from job_description)
# ─────────────────────────────────────────────

# Hard-required skills — these are the things the JD says "you absolutely need"
REQUIRED_SKILLS = [
    "embeddings", "sentence transformers", "bge", "e5", "openai embeddings",
    "vector database", "vector search", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "faiss", "milvus", "dense retrieval", "ann",
    "hybrid search", "bm25",
    "retrieval", "ranking", "information retrieval", "search",
    "ndcg", "mrr", "map", "evaluation", "a/b test", "reranking", "re-ranking",
    "learning to rank", "ltr",
    "nlp", "natural language processing", "text embeddings",
    "python", "pytorch", "tensorflow",
]

# Nice-to-have skills — boost but not required
BONUS_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "xgboost", "gradient boosting", "learning to rank",
    "hr tech", "recruiting", "talent", "marketplace",
    "distributed systems", "inference optimization", "triton", "onnx",
    "open source", "github", "huggingface",
    "rag", "llm", "large language model", "transformer",
    "recommendation", "recommender",
]

# Explicit disqualifiers from the JD
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "persistent systems", "l&t infotech", "ltimindtree",
]

# Industries that are product companies (positive signal)
PRODUCT_INDUSTRIES = [
    "software", "saas", "technology", "fintech", "edtech", "healthtech",
    "e-commerce", "ecommerce", "food delivery", "travel", "gaming",
    "media", "social", "marketplace", "ai", "machine learning",
    "data", "analytics", "cloud", "cybersecurity", "iot",
]

# Target locations (Pune, Noida preferred; NCR cities acceptable)
PREFERRED_LOCATIONS = ["pune", "noida", "delhi", "gurgaon", "gurugram", "ncr", "new delhi"]
ACCEPTABLE_LOCATIONS = ["hyderabad", "bangalore", "bengaluru", "mumbai", "chennai", "india"]

# Full JD text for TF-IDF similarity
JD_TEXT = """
Senior AI Engineer Founding Team Redrob AI Series A AI-native talent intelligence platform
Pune Noida India Hybrid

Production experience embeddings-based retrieval systems sentence-transformers OpenAI embeddings BGE E5
deployed real users embedding drift index refresh retrieval-quality regression production.

Production experience vector databases hybrid search infrastructure Pinecone Weaviate Qdrant Milvus
OpenSearch Elasticsearch FAISS operational experience.

Strong Python code quality.

Hands-on experience designing evaluation frameworks ranking systems NDCG MRR MAP offline-to-online
correlation A/B test interpretation.

LLM fine-tuning LoRA QLoRA PEFT learning-to-rank XGBoost neural ranking.

NLP information retrieval ranking recommendation systems matching search.

Product companies startup experience shipping end-to-end ranking search recommendation real users scale.

Applied ML AI roles product companies not pure services consulting.

5-9 years experience applied ML AI production deployment.

Hybrid retrieval dense retrieval re-ranking LLM integration fine-tune prompt engineering.

Recruiter workflows eval frameworks real engineering shipping.
"""

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def days_since(date_str: Optional[str], ref: date = None) -> Optional[int]:
    if not date_str:
        return None
    if ref is None:
        ref = date.today()
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (ref - d).days
    except Exception:
        return None


def normalize(val: float, lo: float, hi: float) -> float:
    """Clamp and normalize to [0, 1]."""
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def text_lower(s: Optional[str]) -> str:
    return (s or "").lower()


# ─────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────

def is_honeypot(candidate: Dict) -> bool:
    """
    Detect profiles with logically impossible characteristics.
    Honeypots are forced to tier 0 in ground truth — we must avoid them.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    # 1. Check for impossible timelines: experience at company founded after the claimed start
    #    (Proxy: if any role duration is impossibly long relative to total YoE)
    total_yoe = profile.get("years_of_experience", 0) or 0
    total_months_claimed = sum(h.get("duration_months", 0) or 0 for h in career)
    if total_months_claimed > (total_yoe + 2) * 12 + 24:
        # Claims way more experience than total YoE allows
        return True

    # 2. Expert in too many skills with 0 months used
    expert_zero = [s for s in skills
                   if s.get("proficiency") == "expert"
                   and (s.get("duration_months") or 0) == 0]
    if len(expert_zero) >= 3:
        return True

    # 3. Impossibly high assessment scores across too many skills
    assessment = signals.get("skill_assessment_scores") or {}
    perfect_scores = sum(1 for v in assessment.values() if v == 100)
    if perfect_scores >= 5:
        return True

    # 4. Claimed YoE much larger than career history supports
    if career:
        career_start = None
        for h in career:
            try:
                s = datetime.strptime(h["start_date"][:10], "%Y-%m-%d").date()
                if career_start is None or s < career_start:
                    career_start = s
            except Exception:
                pass
        if career_start:
            max_possible_yoe = (date.today() - career_start).days / 365.25
            if total_yoe > max_possible_yoe + 3:
                return True

    # 5. Skills count that's absurdly large (keyword stuffing)
    if len(skills) > 60:
        return True

    return False


# ─────────────────────────────────────────────
# SKILL MATCH SCORE
# ─────────────────────────────────────────────

def skill_match_score(candidate: Dict) -> Tuple[float, float, List[str]]:
    """
    Returns (required_match [0,1], bonus_match [0,1], matched_required_skills).
    Uses candidate skills list + career descriptions for broader matching.
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})

    # Build a combined text blob from the candidate
    skill_names = " ".join(s.get("name", "") for s in skills).lower()
    career_text = " ".join(
        (h.get("description") or "") + " " + (h.get("title") or "")
        for h in career
    ).lower()
    summary = text_lower(profile.get("summary"))
    headline = text_lower(profile.get("headline"))
    all_text = " ".join([skill_names, career_text, summary, headline])

    # Weight skills by proficiency
    proficiency_weight = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.6, "beginner": 0.3}
    skill_set = {}
    for s in skills:
        name = text_lower(s.get("name", ""))
        pw = proficiency_weight.get(s.get("proficiency", ""), 0.5)
        duration_bonus = min(1.0, (s.get("duration_months") or 0) / 24)  # cap at 2yrs
        skill_set[name] = pw * (0.7 + 0.3 * duration_bonus)

    def match_skill_group(skill_group: List[str]) -> Tuple[float, List[str]]:
        matched = []
        total_weight = 0.0
        for target in skill_group:
            # Direct skill match (weighted)
            best = 0.0
            for sname, sweight in skill_set.items():
                if target in sname or sname in target:
                    best = max(best, sweight)
            # Fallback: text mention in career/summary (lower weight)
            if best == 0.0 and target in all_text:
                best = 0.4
            if best > 0:
                matched.append(target)
            total_weight += best
        max_possible = len(skill_group) * 1.0
        return min(1.0, total_weight / max(max_possible, 1)), matched

    req_score, req_matched = match_skill_group(REQUIRED_SKILLS)
    bonus_score, _ = match_skill_group(BONUS_SKILLS)

    return req_score, bonus_score, req_matched


# ─────────────────────────────────────────────
# ROLE FIT SCORE
# ─────────────────────────────────────────────

def role_fit_score(candidate: Dict) -> float:
    """
    Evaluate how well the candidate's role history fits the JD requirements.
    Penalizes: pure consulting background, no product-company experience,
               non-AI roles, pure research roles.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    score = 0.5  # baseline

    # Current title relevance
    title = text_lower(profile.get("current_title", ""))
    ai_titles = ["ml engineer", "ai engineer", "machine learning", "data scientist",
                 "nlp", "research engineer", "applied scientist", "search engineer",
                 "ranking engineer", "recommendation", "retrieval"]
    adjacent_titles = ["software engineer", "backend engineer", "data engineer",
                       "platform engineer", "full stack", "senior engineer"]
    unrelated_titles = ["hr manager", "accountant", "business analyst", "operations",
                        "marketing", "sales", "finance", "customer support",
                        "mechanical", "civil", "content writer", "project manager",
                        "scrum master", "ui/ux", "designer"]

    if any(t in title for t in ai_titles):
        score += 0.25
    elif any(t in title for t in adjacent_titles):
        score += 0.05
    elif any(t in title for t in unrelated_titles):
        score -= 0.30

    # Career history quality
    consulting_roles = 0
    product_roles = 0
    pure_research_roles = 0
    total_roles = len(career)

    for h in career:
        company = text_lower(h.get("company", ""))
        industry = text_lower(h.get("industry", ""))
        role_title = text_lower(h.get("title", ""))
        desc = text_lower(h.get("description", ""))

        # Consulting firms
        if any(cf in company for cf in CONSULTING_FIRMS):
            consulting_roles += 1

        # Product company signal
        if any(pi in industry for pi in PRODUCT_INDUSTRIES):
            product_roles += 1

        # Pure research (academic / research-only)
        if any(w in role_title for w in ["researcher", "phd", "intern", "fellow"]) \
                and "production" not in desc and "deploy" not in desc:
            pure_research_roles += 1

    if total_roles > 0:
        consulting_frac = consulting_roles / total_roles
        product_frac = product_roles / total_roles

        # Penalty for consulting-only background
        if consulting_frac == 1.0 and total_roles >= 2:
            score -= 0.35  # JD explicitly disqualifies this
        elif consulting_frac >= 0.5:
            score -= 0.15

        # Bonus for product company experience
        if product_frac >= 0.5:
            score += 0.20
        elif product_frac > 0:
            score += 0.10

        # Pure research penalty
        if pure_research_roles == total_roles:
            score -= 0.25

    # YoE fit: JD wants 5-9 years; "real" cutoff is production AI experience
    yoe = profile.get("years_of_experience") or 0
    if 5 <= yoe <= 9:
        score += 0.10
    elif 3 <= yoe < 5:
        score += 0.02  # JD says they'd consider this
    elif yoe > 12:
        score -= 0.05  # Possibly over-senior, or title-chaser
    elif yoe < 3:
        score -= 0.20

    return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────
# LOCATION SCORE
# ─────────────────────────────────────────────

def location_score(candidate: Dict) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = text_lower(profile.get("location", ""))
    country = text_lower(profile.get("country", ""))
    will_relocate = signals.get("willing_to_relocate", False)

    if any(loc in location for loc in PREFERRED_LOCATIONS):
        return 1.0
    if any(loc in location for loc in ACCEPTABLE_LOCATIONS) or country == "india":
        return 0.75
    if will_relocate:
        return 0.50
    # Outside India, no relocation
    return 0.20


# ─────────────────────────────────────────────
# BEHAVIORAL AVAILABILITY SCORE
# ─────────────────────────────────────────────

def behavioral_score(candidate: Dict) -> float:
    """
    Measures how "hirable" the candidate actually is right now.
    This is used as a MULTIPLIER, not additive, so a ghosted candidate
    cannot rank in top 10 regardless of skill fit.

    Returns [0, 1] — 1 = highly available, 0 = effectively ghosted.
    """
    sigs = candidate.get("redrob_signals", {})
    today = date.today()

    score_parts = []

    # 1. Recency of last login (stale = not really available)
    days_inactive = days_since(sigs.get("last_active_date"), today)
    if days_inactive is not None:
        if days_inactive <= 14:
            recency = 1.0
        elif days_inactive <= 30:
            recency = 0.85
        elif days_inactive <= 60:
            recency = 0.65
        elif days_inactive <= 120:
            recency = 0.40
        elif days_inactive <= 180:
            recency = 0.20
        else:
            recency = 0.05
        score_parts.append(("recency", recency, 0.30))

    # 2. Open to work
    open_to_work = 1.0 if sigs.get("open_to_work_flag") else 0.3
    score_parts.append(("open_to_work", open_to_work, 0.15))

    # 3. Recruiter response rate
    rr = sigs.get("recruiter_response_rate")
    if rr is not None:
        score_parts.append(("response_rate", float(rr), 0.20))

    # 4. Notice period (JD prefers <30d; accepts up to ~90d)
    notice = sigs.get("notice_period_days")
    if notice is not None:
        if notice <= 15:
            np_score = 1.0
        elif notice <= 30:
            np_score = 0.90
        elif notice <= 60:
            np_score = 0.70
        elif notice <= 90:
            np_score = 0.50
        else:
            np_score = 0.25
        score_parts.append(("notice", np_score, 0.10))

    # 5. Interview completion rate
    icr = sigs.get("interview_completion_rate")
    if icr is not None:
        score_parts.append(("interview_completion", float(icr), 0.10))

    # 6. Offer acceptance rate (if available)
    oar = sigs.get("offer_acceptance_rate")
    if oar is not None and oar >= 0:
        score_parts.append(("offer_acceptance", float(oar), 0.05))

    # 7. Applications submitted in last 30d (actively job-hunting)
    apps = sigs.get("applications_submitted_30d") or 0
    apps_score = min(1.0, apps / 5)
    score_parts.append(("active_search", apps_score, 0.10))

    if not score_parts:
        return 0.5

    total_weight = sum(w for _, _, w in score_parts)
    weighted_sum = sum(v * w for _, v, w in score_parts)
    return weighted_sum / total_weight


# ─────────────────────────────────────────────
# CAREER QUALITY SCORE
# ─────────────────────────────────────────────

def career_quality_score(candidate: Dict) -> float:
    """
    Signal quality: GitHub activity, endorsements, assessments, education.
    JD values external validation (open source, talks, papers).
    """
    sigs = candidate.get("redrob_signals", {})
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})

    score = 0.5

    # GitHub activity (JD values external validation)
    gh = sigs.get("github_activity_score", -1)
    if gh == -1:
        score -= 0.05  # no GitHub linked
    elif gh >= 70:
        score += 0.20
    elif gh >= 40:
        score += 0.12
    elif gh >= 15:
        score += 0.05

    # Skill assessment scores (verified proficiency)
    assessments = sigs.get("skill_assessment_scores") or {}
    if assessments:
        avg_assessment = sum(assessments.values()) / len(assessments)
        score += normalize(avg_assessment, 30, 90) * 0.15

    # Endorsements per skill (social proof)
    total_endorsements = sigs.get("endorsements_received") or 0
    n_skills = max(len(skills), 1)
    endorsements_per_skill = total_endorsements / n_skills
    score += normalize(endorsements_per_skill, 0, 30) * 0.10

    # Education tier
    best_tier = None
    for e in education:
        tier = e.get("tier", "tier_4")
        tier_num = int(tier.split("_")[1]) if "_" in tier else 4
        if best_tier is None or tier_num < best_tier:
            best_tier = tier_num
    if best_tier == 1:
        score += 0.10
    elif best_tier == 2:
        score += 0.06
    elif best_tier == 3:
        score += 0.02

    # Profile completeness
    completeness = sigs.get("profile_completeness_score", 0) or 0
    score += normalize(completeness, 50, 100) * 0.08

    return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────
# TFIDF TEXT SIMILARITY (fast, no GPU needed)
# ─────────────────────────────────────────────

def build_candidate_text(candidate: Dict) -> str:
    """Build a rich text representation of a candidate for TF-IDF."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    certs = candidate.get("certifications", [])

    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        " ".join(s.get("name", "") for s in skills),
        " ".join(h.get("title", "") + " " + (h.get("description") or "") for h in career),
        " ".join(c.get("name", "") for c in (certs or [])),
    ]
    return " ".join(p for p in parts if p)


def compute_tfidf_scores(candidates: List[Dict]) -> np.ndarray:
    """Return cosine similarity of each candidate to the JD."""
    print(f"  Building TF-IDF corpus for {len(candidates)} candidates...", flush=True)
    corpus = [build_candidate_text(c) for c in candidates]
    corpus_with_jd = [JD_TEXT] + corpus

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    matrix = vectorizer.fit_transform(corpus_with_jd)
    jd_vec = matrix[0]
    candidate_matrix = matrix[1:]
    similarities = cosine_similarity(jd_vec, candidate_matrix).flatten()
    return similarities


# ─────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────

WEIGHTS = {
    "tfidf":          0.15,  # text semantic match to JD
    "skill_required": 0.30,  # required skill match
    "skill_bonus":    0.08,  # bonus skills
    "role_fit":       0.22,  # role/career quality
    "location":       0.08,  # location fit
    "career_quality": 0.17,  # github, assessments, endorsements, education
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 0.01, "Weights must sum to 1"


def composite_score(
    candidate: Dict,
    tfidf_score: float,
    honeypot: bool,
) -> Tuple[float, Dict]:
    """
    Compute final score for a candidate.
    behavioral_score acts as a multiplier so unavailable candidates can't
    artificially inflate their ranking.
    """
    if honeypot:
        return 0.0, {"honeypot": True}

    req_skill, bonus_skill, matched_skills = skill_match_score(candidate)
    role = role_fit_score(candidate)
    location = location_score(candidate)
    career = career_quality_score(candidate)
    behavioral = behavioral_score(candidate)

    base = (
        WEIGHTS["tfidf"] * tfidf_score
        + WEIGHTS["skill_required"] * req_skill
        + WEIGHTS["skill_bonus"] * bonus_skill
        + WEIGHTS["role_fit"] * role
        + WEIGHTS["location"] * location
        + WEIGHTS["career_quality"] * career
    )

    # Behavioral multiplier: scales base by [0.05, 1.0]
    # A ghosted candidate (behavioral=0.1) gets base * 0.12 — can't win top 10
    behavioral_multiplier = 0.05 + 0.95 * behavioral
    final = base * behavioral_multiplier

    details = {
        "tfidf": round(tfidf_score, 4),
        "skill_required": round(req_skill, 4),
        "skill_bonus": round(bonus_skill, 4),
        "role_fit": round(role, 4),
        "location": round(location, 4),
        "career_quality": round(career, 4),
        "behavioral": round(behavioral, 4),
        "base": round(base, 4),
        "final": round(final, 4),
        "matched_skills": matched_skills[:6],
    }
    return final, details


# ─────────────────────────────────────────────
# REASONING GENERATOR
# ─────────────────────────────────────────────

def generate_reasoning(candidate: Dict, details: Dict, rank: int) -> str:
    """Generate specific, honest, rank-consistent reasoning for a candidate."""
    profile = candidate.get("profile", {})
    sigs = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "")

    # Top skill names (AI-relevant ones)
    ai_skill_names = [
        s["name"] for s in skills
        if any(kw in s["name"].lower() for kw in [
            "nlp", "embedding", "vector", "retrieval", "search", "llm", "rag",
            "pytorch", "tensorflow", "sklearn", "faiss", "pinecone", "weaviate",
            "bert", "transformer", "fine-tun", "ranking", "recommendation",
            "elasticsearch", "opensearch", "bm25", "qdrant",
        ])
    ][:4]

    response_rate = sigs.get("recruiter_response_rate")
    notice = sigs.get("notice_period_days")
    last_active = sigs.get("last_active_date", "")
    days_inactive = days_since(last_active)
    open_to_work = sigs.get("open_to_work_flag")
    gh = sigs.get("github_activity_score", -1)

    # Build reasoning components
    parts = []

    # Who they are
    parts.append(f"{title} at {company} with {yoe:.1f} years' experience")

    # Skill signal
    if ai_skill_names:
        parts.append(f"relevant AI/retrieval skills: {', '.join(ai_skill_names)}")
    elif details.get("skill_required", 0) > 0.3:
        matched = details.get("matched_skills", [])
        if matched:
            parts.append(f"partial skill match ({', '.join(matched[:3])})")

    # Location
    if location:
        parts.append(f"based in {location}")

    # Behavioral honesty
    concerns = []
    if days_inactive and days_inactive > 90:
        concerns.append(f"inactive for {days_inactive} days")
    if response_rate is not None and response_rate < 0.25:
        concerns.append(f"low recruiter response rate ({response_rate:.0%})")
    if notice and notice > 60:
        concerns.append(f"long notice period ({notice} days)")
    if not open_to_work:
        concerns.append("not marked open to work")

    if concerns:
        parts.append("concern: " + "; ".join(concerns))
    elif rank <= 10 and gh > 50:
        parts.append(f"strong GitHub activity (score {gh:.0f})")

    return "; ".join(parts) + "."


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def load_candidates(path: str) -> List[Dict]:
    print(f"Loading candidates from {path}...", flush=True)
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
            if (i + 1) % 10000 == 0:
                print(f"  Loaded {i + 1} candidates...", flush=True)
    print(f"  Total: {len(candidates)} candidates loaded.", flush=True)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=100, help="Number of candidates to output")
    args = parser.parse_args()

    t_start = datetime.now()
    print(f"\n{'='*60}")
    print("Redrob Hackathon Ranker")
    print(f"Started: {t_start.strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    # Step 1: Load
    candidates = load_candidates(args.candidates)

    # Step 2: Honeypot detection
    print("\nRunning honeypot detection...", flush=True)
    honeypots = {c["candidate_id"]: is_honeypot(c) for c in candidates}
    n_honeypots = sum(honeypots.values())
    print(f"  Detected {n_honeypots} likely honeypots (will score 0.0)", flush=True)

    # Step 3: TF-IDF similarity
    print("\nComputing TF-IDF similarity to JD...", flush=True)
    tfidf_scores = compute_tfidf_scores(candidates)
    print("  Done.", flush=True)

    # Step 4: Score all candidates
    print("\nScoring all candidates...", flush=True)
    results = []
    for i, c in enumerate(candidates):
        cid = c["candidate_id"]
        is_hp = honeypots.get(cid, False)
        score, details = composite_score(c, tfidf_scores[i], is_hp)
        results.append((cid, score, details, c))
        if (i + 1) % 10000 == 0:
            print(f"  Scored {i + 1}/{len(candidates)}...", flush=True)

    # Step 5: Sort and take top N
    results.sort(key=lambda x: x[1], reverse=True)
    top_results = results[:args.top_n]

    # Step 6: Write CSV
    print(f"\nWriting top {args.top_n} to {args.out}...", flush=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cid, score, details, candidate) in enumerate(top_results, start=1):
            reasoning = generate_reasoning(candidate, details, rank)
            writer.writerow([cid, rank, round(score, 6), reasoning])

    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()

    print(f"\n{'='*60}")
    print(f"Done! Elapsed: {elapsed:.1f}s")
    print(f"Output: {args.out}")
    print(f"\nTop 5 candidates:")
    for rank, (cid, score, details, candidate) in enumerate(top_results[:5], 1):
        p = candidate.get("profile", {})
        print(f"  {rank}. {cid} | {p.get('current_title')} @ {p.get('current_company')} "
              f"| score={score:.4f} | behavioral={details.get('behavioral', '?'):.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
