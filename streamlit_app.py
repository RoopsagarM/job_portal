# streamlit_app.py
import os
import re
from typing import List, Dict
import requests
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader


# =========================
# PDF → text
# =========================
def pdf_to_text(file) -> str:
    try:
        reader = PdfReader(file)
        chunks = []
        for p in reader.pages:
            try:
                chunks.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(chunks).strip()
    except Exception:
        return ""


# =========================
# Experience detection
# =========================
YR_RANGE = re.compile(r"(\d{1,2})\s*-\s*(\d{1,2})\s*(?:years|yrs)\b", re.I)
YR_OFEXP = re.compile(r"(\d{1,2})\s*\+?\s*(?:years|yrs)\s+of\s+experience", re.I)
YR_GENERIC = re.compile(r"(?<!-)\b(\d{1,2})\s*\+?\s*(?:years|yrs)\b", re.I)
MON_GENERIC = re.compile(r"(\d{1,2})\s*[- ]?(?:months|mos|month)\b", re.I)

def parse_experience_years(text: str) -> float:
    text = text.lower()
    vals = []

    for m in YR_RANGE.finditer(text):
        a, b = int(m.group(1)), int(m.group(2)); vals.append(max(a, b))
    for m in YR_OFEXP.finditer(text):
        vals.append(int(m.group(1)))
    for m in YR_GENERIC.finditer(text):
        vals.append(int(m.group(1)))
    for m in MON_GENERIC.finditer(text):
        vals.append(int(m.group(1)) / 12.0)

    return float(max(vals)) if vals else 0.0

def seniority_from_years(years: float) -> str:
    if years <= 1.5: return "entry"
    if years <= 5:   return "mid"
    return "senior"


# =========================
# Keyword extraction (TF-IDF)
# =========================
def extract_keywords(text: str, top_n: int = 12) -> List[str]:
    if not text.strip():
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform([text])
    scores = np.asarray(X.todense()).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = scores.argsort()[::-1]
    seen, out = set(), []
    for i in order:
        t = terms[i].strip()
        if t and t not in seen:
            seen.add(t); out.append(t)
        if len(out) >= top_n:
            break
    return out

def build_smart_query(keywords: List[str]) -> str:
    text = " ".join(keywords).lower()
    query = "software engineer OR developer"
    if any(k in text for k in ["data","sql","python","pandas","analysis","ml","ai","analytics"]):
        query = "data analyst OR data engineer OR machine learning engineer OR data scientist"
    elif any(k in text for k in ["aws","cloud","distributed","redshift","snowflake","devops","docker","kubernetes"]):
        query = "cloud engineer OR backend engineer OR devops engineer OR software engineer"
    elif any(k in text for k in ["react","javascript","frontend","ui","web","typescript"]):
        query = "frontend developer OR full stack engineer OR web developer"
    return query


# =========================
# Fetch jobs (RapidAPI → JSearch)
# =========================
def fetch_jobs(
    query: str,
    days: int = 7,
    pages: int = 2,
    country: str = "us",
    remote_only: bool = True,
    max_jobs: int = 60,
) -> List[Dict]:
    url = "https://jsearch.p.rapidapi.com/search"
    date_posted = "month" if days >= 30 else ("week" if days >= 7 else ("3days" if days >= 3 else "today"))
    params = {
        "query": query,
        "page": 1,
        "num_pages": pages,
        "date_posted": date_posted,
        "country": country,
        "remote_only": "true" if remote_only else "false",
    }

    # 🔑 Hardcoded API key here
    api_key = "f3bd6b71a8mshf338bb20bddde4fp1f4e9ajsnf9be665c5554"

    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}

    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Error fetching jobs: {r.text[:300]}")
        return []
    return (r.json() or {}).get("data", [])[:max_jobs]


# =========================
# Rank by resume fit (TF-IDF + cosine)
# =========================
def rank_jobs(resume_text: str, jobs: List[Dict], top_k: int = 30) -> pd.DataFrame:
    if not jobs: return pd.DataFrame()
    docs = [resume_text]
    titles, companies, locations, dates, links, descs = [], [], [], [], [], []

    for j in jobs:
        desc = j.get("job_description") or j.get("description") or ""
        if not desc:
            desc = f"{j.get('job_title','')} {j.get('category','')}"
        docs.append(desc)

        titles.append(j.get("job_title", ""))
        companies.append(j.get("employer_name", ""))
        city = j.get("job_city") or j.get("job_location") or ""
        region = j.get("job_state") or ""
        country = j.get("job_country") or ""
        loc = ", ".join([p for p in [city, region, country] if p])
        locations.append(loc)
        dates.append(j.get("job_posted_at_datetime_utc") or j.get("job_posted_at") or "")
        link = j.get("job_apply_link") or (j.get("job_apply_links",[{}])[0].get("link") if j.get("job_apply_links") else None) or j.get("job_url")
        links.append(link)
        descs.append(desc)

    vec = TfidfVectorizer(stop_words="english", max_df=0.9)
    X = vec.fit_transform(docs)
    sim = cosine_similarity(X[0:1], X[1:]).ravel()

    df = pd.DataFrame({
        "score": sim,
        "title": titles,
        "company": companies,
        "location": locations,
        "posted_utc": dates,
        "apply_url": links,
        "description": descs,
    }).sort_values("score", ascending=False)

    return df.head(top_k).reset_index(drop=True)


# =========================
# Title-level filtering by seniority
# =========================
SENIOR_WORDS = ["senior","lead","principal","staff","architect","manager","head"]
ENTRY_WORDS  = ["junior","entry","intern","new grad","graduate","associate","assoc"]

def title_matches_level(title: str, level: str) -> bool:
    t = (title or "").lower()
    if not t: return True
    if level == "senior":
        return any(w in t for w in SENIOR_WORDS)
    if level == "entry":
        if any(w in t for w in SENIOR_WORDS): return False
        return True
    if any(w in t for w in SENIOR_WORDS) or any(w in t for w in ENTRY_WORDS):
        return False
    return True


# =========================
# UI
# =========================
st.set_page_config(page_title="Resume → Fresh Job Finder", page_icon="🔎", layout="wide")
st.markdown(
    "<h1 style='margin:0 0 8px 0'>🔎 Resume → Fresh Job Finder</h1>"
    "<p style='color:#bbb;margin:0'>Upload your resume → auto keywords → detect experience → fetch recent jobs → ranked & filtered</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Configure")
    days = st.selectbox("Jobs from last…", [1,3,7,30], index=2)
    country = st.selectbox("Country", ["us","ca","in","gb","de","au"], index=0)
    pages = st.slider("Pages to fetch (≈10 each)", 1, 5, 2)
    remote_only = st.toggle("Remote only", value=True)

st.subheader("Upload your resume (PDF)")
file = st.file_uploader("Drag & drop or browse…", type=["pdf"], label_visibility="collapsed")

if file and st.button("Find jobs", type="primary"):
    resume_text = pdf_to_text(file)
    if not resume_text:
        st.error("Couldn’t read text from PDF. Try another file.")
        st.stop()

    years = parse_experience_years(resume_text)
    level = seniority_from_years(years)
    st.info(f"Detected experience: ~{years:.1f} years • Seniority: {level.title()}")

    with st.status("Extracting keywords…", state="running"):
        kws = extract_keywords(resume_text, top_n=12)
        st.write("**Keywords:**", ", ".join(kws) if kws else "(none)")

    query = build_smart_query(kws)
    st.write("**Query:**", query)

    with st.status("Fetching jobs…", state="running"):
        jobs = fetch_jobs(query, days=days, pages=pages, country=country, remote_only=remote_only, max_jobs=60)
        jobs = [j for j in jobs if title_matches_level(j.get("job_title",""), level)]
        st.write(f"Fetched {len(jobs)} jobs after seniority filter.")

    if not jobs:
        st.warning("No jobs found. Try increasing days/pages, switching country, or disabling remote-only.")
        st.stop()

    with st.status("Ranking by resume match…", state="running"):
        ranked = rank_jobs(resume_text, jobs, top_k=30)

    st.success(f"Top {len(ranked)} matches")
    for i, row in ranked.iterrows():
        st.markdown(f"### {i+1}. {row['title']}")
        st.write(f"**Company:** {row['company']}  \n**Location:** {row['location'] or '—'}")
        if row["posted_utc"]:
            st.caption(f"Posted: {row['posted_utc']}")
        st.progress(float(max(0.0, min(1.0, row['score']))))
        c1, c2 = st.columns([0.18, 0.82])
        with c1:
            if row["apply_url"]:
                st.link_button("Apply", row["apply_url"])
        with c2:
            with st.expander("Description"):
                st.write((row["description"] or "")[:2000])

st.divider()
with st.expander("Tips"):
    st.markdown(
        "- API key is **hardcoded** into this app.\n"
        "- Increase *days* or *pages* to see more roles.\n"
        "- Experience parsing understands **years** and **months** (e.g., “6 months”)."
    )
