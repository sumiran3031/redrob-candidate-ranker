import streamlit as st
import json
import csv
import io
import math
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="ByteBesties — Redrob Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Redrob Candidate Ranker")
st.markdown("**Team ByteBesties** — Upload a JSONL file of candidates, get the top 100 ranked.")
st.markdown("---")

# ── paste the entire JD_TEXT, REQUIRED_SKILLS, BONUS_SKILLS,
#    CONSULTING_FIRMS, PRODUCT_INDUSTRIES, PREFERRED_LOCATIONS,
#    ACCEPTABLE_LOCATIONS constants from rank.py here ──

JD_TEXT = """
Senior AI Engineer Founding Team Redrob AI Series A AI-native talent intelligence platform
Pune Noida India Hybrid Production experience embeddings-based retrieval systems
sentence-transformers OpenAI embeddings BGE E5 deployed real users embedding drift
index refresh retrieval-quality regression production. Production experience vector databases
hybrid search infrastructure Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS
operational experience. Strong Python code quality. Hands-on experience designing evaluation
frameworks ranking systems NDCG MRR MAP offline-to-online correlation A/B test interpretation.
LLM fine-tuning LoRA QLoRA PEFT learning-to-rank XGBoost neural ranking. NLP information
retrieval ranking recommendation systems matching search. Product companies startup experience
shipping end-to-end ranking search recommendation real users scale. Applied ML AI roles product
companies not pure services consulting. 5-9 years experience applied ML AI production deployment.
"""

REQUIRED_SKILLS = [
    "embeddings","sentence transformers","bge","e5","openai embeddings",
    "vector database","vector search","pinecone","weaviate","qdrant","milvus",
    "opensearch","elasticsearch","faiss","dense retrieval","ann","hybrid search","bm25",
    "retrieval","ranking","information retrieval","search","ndcg","mrr","map",
    "evaluation","reranking","re-ranking","learning to rank","ltr",
    "nlp","natural language processing","text embeddings","python","pytorch","tensorflow",
]
BONUS_SKILLS = [
    "lora","qlora","peft","fine-tuning","fine tuning","finetuning",
    "xgboost","gradient boosting","learning to rank","hr tech","recruiting",
    "distributed systems","rag","llm","large language model","transformer",
    "recommendation","recommender","open source","github","huggingface",
]
CONSULTING_FIRMS = [
    "tcs","tata consultancy","infosys","wipro","accenture",
    "cognizant","capgemini","hcl","tech mahindra","mphasis",
]
PRODUCT_INDUSTRIES = [
    "software","saas","technology","fintech","edtech","healthtech",
    "e-commerce","ecommerce","food delivery","travel","gaming",
    "media","social","marketplace","ai","machine learning","data","analytics","cloud",
]
PREFERRED_LOCATIONS = ["pune","noida","delhi","gurgaon","gurugram","ncr","new delhi"]
ACCEPTABLE_LOCATIONS = ["hyderabad","bangalore","bengaluru","mumbai","chennai","india"]

WEIGHTS = {
    "tfidf":          0.15,
    "skill_required": 0.30,
    "skill_bonus":    0.08,
    "role_fit":       0.22,
    "location":       0.12,
    "career_quality": 0.13,
}

def days_since(date_str, ref=None):
    if not date_str: return None
    if ref is None: ref = date.today()
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (ref - d).days
    except: return None

def normalize(val, lo, hi):
    if hi == lo: return 0.5
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))

def text_lower(s): return (s or "").lower()

def is_honeypot(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    total_yoe = profile.get("years_of_experience", 0) or 0
    total_months = sum(h.get("duration_months", 0) or 0 for h in career)
    if total_months > (total_yoe + 2) * 12 + 24: return True
    expert_zero = [s for s in skills if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) == 0]
    if len(expert_zero) >= 3: return True
    assessment = signals.get("skill_assessment_scores") or {}
    if sum(1 for v in assessment.values() if v == 100) >= 5: return True
    if len(skills) > 60: return True
    return False

def skill_match_score(candidate):
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    skill_names = " ".join(s.get("name", "") for s in skills).lower()
    career_text = " ".join((h.get("description") or "") + " " + (h.get("title") or "") for h in career).lower()
    all_text = " ".join([skill_names, career_text, text_lower(profile.get("summary")), text_lower(profile.get("headline"))])
    pw_map = {"expert": 1.0, "advanced": 0.85, "intermediate": 0.6, "beginner": 0.3}
    skill_set = {}
    for s in skills:
        name = text_lower(s.get("name", ""))
        pw = pw_map.get(s.get("proficiency", ""), 0.5)
        dur = min(1.0, (s.get("duration_months") or 0) / 24)
        skill_set[name] = pw * (0.7 + 0.3 * dur)
    def match(group):
        matched, total = [], 0.0
        for target in group:
            best = 0.0
            for sname, sw in skill_set.items():
                if target in sname or sname in target: best = max(best, sw)
            if best == 0.0 and target in all_text: best = 0.4
            if best > 0: matched.append(target)
            total += best
        return min(1.0, total / max(len(group), 1)), matched
    req_score, req_matched = match(REQUIRED_SKILLS)
    bonus_score, _ = match(BONUS_SKILLS)
    return req_score, bonus_score, req_matched

def role_fit_score(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    score = 0.5
    title = text_lower(profile.get("current_title", ""))
    ai_titles = ["ml engineer","ai engineer","machine learning","data scientist","nlp","research engineer","applied scientist","search engineer","ranking engineer","recommendation","retrieval"]
    unrelated = ["hr manager","accountant","business analyst","operations","marketing","sales","finance","customer support","content writer","project manager","scrum master","designer"]
    if any(t in title for t in ai_titles): score += 0.25
    elif any(t in title for t in unrelated): score -= 0.30
    consulting_roles = sum(1 for h in career if any(cf in text_lower(h.get("company","")) for cf in CONSULTING_FIRMS))
    product_roles = sum(1 for h in career if any(pi in text_lower(h.get("industry","")) for pi in PRODUCT_INDUSTRIES))
    total = len(career)
    if total > 0:
        if consulting_roles / total == 1.0 and total >= 2: score -= 0.35
        elif consulting_roles / total >= 0.5: score -= 0.15
        if product_roles / total >= 0.5: score += 0.20
        elif product_roles > 0: score += 0.10
    yoe = profile.get("years_of_experience") or 0
    if 5 <= yoe <= 9: score += 0.10
    elif yoe < 3: score -= 0.20
    return max(0.0, min(1.0, score))

def location_score(candidate):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    location = text_lower(profile.get("location", ""))
    country = text_lower(profile.get("country", ""))
    if any(loc in location for loc in PREFERRED_LOCATIONS): return 1.0
    if any(loc in location for loc in ACCEPTABLE_LOCATIONS) or country == "india": return 0.75
    if signals.get("willing_to_relocate"): return 0.50
    return 0.20

def behavioral_score(candidate):
    sigs = candidate.get("redrob_signals", {})
    today = date.today()
    parts = []
    days_inactive = days_since(sigs.get("last_active_date"), today)
    if days_inactive is not None:
        if days_inactive <= 14: rec = 1.0
        elif days_inactive <= 30: rec = 0.85
        elif days_inactive <= 60: rec = 0.65
        elif days_inactive <= 120: rec = 0.40
        elif days_inactive <= 180: rec = 0.20
        else: rec = 0.05
        parts.append(rec * 0.30)
    parts.append((1.0 if sigs.get("open_to_work_flag") else 0.3) * 0.15)
    rr = sigs.get("recruiter_response_rate")
    if rr is not None: parts.append(float(rr) * 0.20)
    notice = sigs.get("notice_period_days")
    if notice is not None:
        np_s = 1.0 if notice<=15 else 0.90 if notice<=30 else 0.70 if notice<=60 else 0.50 if notice<=90 else 0.25
        parts.append(np_s * 0.10)
    icr = sigs.get("interview_completion_rate")
    if icr is not None: parts.append(float(icr) * 0.10)
    apps = sigs.get("applications_submitted_30d") or 0
    parts.append(min(1.0, apps / 5) * 0.10)
    return sum(parts) / 0.95 if parts else 0.5

def career_quality_score(candidate):
    sigs = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    education = candidate.get("education", [])
    score = 0.5
    gh = sigs.get("github_activity_score", -1)
    if gh == -1: score -= 0.05
    elif gh >= 70: score += 0.20
    elif gh >= 40: score += 0.12
    elif gh >= 15: score += 0.05
    assessments = sigs.get("skill_assessment_scores") or {}
    if assessments: score += normalize(sum(assessments.values())/len(assessments), 30, 90) * 0.15
    total_endorsements = sigs.get("endorsements_received") or 0
    score += normalize(total_endorsements / max(len(skills), 1), 0, 30) * 0.10
    best_tier = None
    for e in education:
        tier = e.get("tier", "tier_4")
        t = int(tier.split("_")[1]) if "_" in tier else 4
        if best_tier is None or t < best_tier: best_tier = t
    if best_tier == 1: score += 0.10
    elif best_tier == 2: score += 0.06
    elif best_tier == 3: score += 0.02
    return max(0.0, min(1.0, score))

def build_text(candidate):
    p = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    return " ".join(filter(None, [
        p.get("headline",""), p.get("summary",""),
        " ".join(s.get("name","") for s in skills),
        " ".join((h.get("title","") + " " + (h.get("description") or "")) for h in career),
    ]))

def score_candidate(candidate, tfidf_score, honeypot):
    if honeypot: return 0.0
    req, bonus, _ = skill_match_score(candidate)
    role = role_fit_score(candidate)
    loc = location_score(candidate)
    cq = career_quality_score(candidate)
    beh = behavioral_score(candidate)
    base = (WEIGHTS["tfidf"]*tfidf_score + WEIGHTS["skill_required"]*req +
            WEIGHTS["skill_bonus"]*bonus + WEIGHTS["role_fit"]*role +
            WEIGHTS["location"]*loc + WEIGHTS["career_quality"]*cq)
    return base * (0.05 + 0.95 * beh)

def run_ranker(candidates):
    corpus = [JD_TEXT] + [build_text(c) for c in candidates]
    vec = TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True, min_df=2)
    matrix = vec.fit_transform(corpus)
    sims = cosine_similarity(matrix[0], matrix[1:]).flatten()
    results = []
    for i, c in enumerate(candidates):
        hp = is_honeypot(c)
        s = score_candidate(c, sims[i], hp)
        results.append((c["candidate_id"], s, c))
    results.sort(key=lambda x: -x[1])
    return results[:100]

# ── UI ──
col1, col2 = st.columns([2, 1])
with col1:
    uploaded = st.file_uploader(
        "Upload candidates JSONL file (max ~5MB for demo)",
        type=["jsonl", "json"]
    )
with col2:
    st.info("**How to use:**\n1. Upload a JSONL file\n2. Each line = one candidate JSON\n3. Download the ranked CSV")

if uploaded:
    with st.spinner("Loading candidates..."):
        content = uploaded.read().decode("utf-8")
        # Handle both JSON array and JSONL formats
content_stripped = content.strip()
if content_stripped.startswith("["):
    # It's a JSON array (sample_candidates.json)
    candidates = json.loads(content_stripped)
else:
    # It's JSONL (one candidate per line)
    content = content.strip()
        if content.startswith("["):
            candidates = json.loads(content)
        else:
            lines = [l.strip() for l in content.split("\n") if l.strip()]
            candidates = [json.loads(l) for l in lines]
    st.success(f"Loaded {len(candidates)} candidates")

    if st.button("🚀 Run Ranker", type="primary"):
        with st.spinner(f"Ranking {len(candidates)} candidates... this takes ~10s per 5000 candidates"):
            top100 = run_ranker(candidates)

        st.success(f"Done! Showing top {min(len(top100), 100)} candidates.")

        # Build display dataframe
        rows = []
        for rank, (cid, score, c) in enumerate(top100, 1):
            p = c.get("profile", {})
            sigs = c.get("redrob_signals", {})
            rows.append({
                "Rank": rank,
                "Candidate ID": cid,
                "Title": p.get("current_title", ""),
                "Company": p.get("current_company", ""),
                "Location": p.get("location", ""),
                "YoE": p.get("years_of_experience", ""),
                "Score": round(score, 4),
                "Behavioral": round(behavioral_score(c), 2),
                "Open to Work": "✅" if sigs.get("open_to_work_flag") else "❌",
            })

        import pandas as pd
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=600)

        # Download CSV
        csv_rows = []
        for rank, (cid, score, c) in enumerate(top100, 1):
            p = c.get("profile", {})
            sigs = c.get("redrob_signals", {})
            skills = [s["name"] for s in c.get("skills",[]) if any(
                kw in s["name"].lower() for kw in ["embedding","vector","retrieval","search","nlp","llm","rag","faiss","pinecone"]
            )][:4]
            reasoning = f"{p.get('current_title')} at {p.get('current_company')} with {p.get('years_of_experience')} yrs; skills: {', '.join(skills)}; location: {p.get('location')}."
            csv_rows.append(f'{cid},{rank},{round(score,6)},"{reasoning}"')

        csv_content = "candidate_id,rank,score,reasoning\n" + "\n".join(csv_rows)
        st.download_button(
            "⬇️ Download submission.csv",
            data=csv_content,
            file_name="submission.csv",
            mime="text/csv"
        )
else:
    st.markdown("### 👆 Upload a JSONL file to get started")
    st.markdown("The ranker scores candidates against the Redrob Senior AI Engineer JD using TF-IDF + behavioral signals.")