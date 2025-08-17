# streamlit_app.py
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import streamlit as st
import requests
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader


# ----------------------------
# Utility: PDF -> text
# ----------------------------
def pdf_to_text(file) -> str:
    try:
        reader = PdfReader(file)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(texts).strip()
    except Exception:
        return ""


# ----------------------------
# TF-IDF keyword extraction
# ----------------------------
def extract_keywords(text: str, top_n: int = 12) -> List[str]:
    if not text or not text.strip():
        return []
    # simple TF-IDF on a single-document bag to get top terms
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform([text])
    scores = np.asarray(X.todense()).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = scores.argsort()[::-1]
    # de-dup while preserving order
    seen, result = set(), []
    for idx in order:
        term = terms[idx].strip()
        if term and term not in seen:
            seen.add(term)
            result.append(term)
        if len(result) >= top_n:
            break
    return result


def build_query(keywords: List[str]) -> str:
    # jsearch likes space-delimited ORs
    safe = [k.replace('"', "") for k in keywords]
    return " OR ".join(safe)


# ----------------------------
# Fetch jobs via RapidAPI -> JSearch
# ----------------------------
def fetch_jobs(
    query: str,
    days: int = 7,
    pages: int = 1,
    country: str = "us",
    remote_only: bool = False,
    rapid_key: str = "",
    rapid_host: str = "jsearch.p.rapidapi.com",
    max_jobs: int = 40,
) -> List[Dict]:
    url = "https://jsearch.p.rapidapi.com/search"
    params = {
        "query": query,
        "page": 1,
        "num_pages": pages,
        "date_posted": "today" if days == 1 else ("3days" if days <= 3 else ("week" if days <= 7 else "month")),
        "country": country,
        "remote_only": "true" if remote_only else "false",
    }
    headers = {"X-RapidAPI-Key": rapid_key, "X-RapidAPI-Host": rapid_host}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        items = data.get("data", []) or []
        return items[:max_jobs]
    except Exception as e:
        st.error(f"Error fetching jobs: {e}")
        return []


# ----------------------------
# Rank jobs vs resume (TF-IDF + cosine)
# ----------------------------
def rank_jobs(resume_text: str, jobs: List[Dict], top_k: int = 30) -> pd.DataFrame:
    if not jobs:
        return pd.DataFrame()
    docs = [resume_text]
    titles, companies, locations, dates, links, descs = [], [], [], [], [], []

    for j in jobs:
        desc = j.get("job_description") or j.get("description") or ""
        # if desc empty, use title + category
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
        links.append(j.get("job_apply_link") or j.get("job_apply_links", [{}])[0].get("link") if j.get("job_apply_links") else j.get("job_url"))
        descs.append(desc)

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
    X = vectorizer.fit_transform(docs)
    sim = cosine_similarity(X[0:1], X[1:]).ravel()  # similarity resume vs each job

    df = pd.DataFrame(
        {
            "score": sim,
            "title": titles,
            "company": companies,
            "location": locations,
            "posted_utc": dates,
            "apply_url": links,
            "description": descs,
        }
    ).sort_values("score", ascending=False)

    return df.head(top_k).reset_index(drop=True)


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Resume → Fresh Job Finder", page_icon="🔎", layout="wide")

st.markdown(
    "<h1 style='margin-bottom:0'>🔎 Resume → Fresh Job Finder</h1>"
    "<p style='color:#bbb;margin-top:4px'>Upload your resume → auto keywords → fetch recent jobs → ranked by fit</p>",
    unsafe_allow_html=True,
)

# Sidebar config
with st.sidebar:
    st.header("⚙️ Configure")
    days = st.selectbox("Show jobs from the last…", [1, 3, 7, 30], index=2)
    country = st.selectbox("Country", ["us", "ca", "in", "gb", "de", "au"], index=0)
    pages = st.slider("Pages to fetch (each ≈10 jobs)", 1, 5, 2)
    remote_only = st.toggle("Remote only", value=False)

    st.divider()
    # Secrets first, then manual
    api_key = st.secrets.get("f3bd6b71a8mshf338bb20bddde4fp1f4e9ajsnf9be665c5554", "")
    host = st.secrets.get("RAPIDAPI_HOST", "jsearch.p.rapidapi.com")
    if not api_key:
        api_key = st.text_input("RapidAPI Key (securely add in Secrets later)", type="password")
    host = st.text_input("RapidAPI Host", value=host)

st.subheader("Upload your resume (PDF)")
file = st.file_uploader("Drag & drop or Browse…", type=["pdf"], label_visibility="collapsed")

if st.button("Find jobs", type="primary", disabled=not file or not api_key):
    if not api_key:
        st.error("Missing RapidAPI key. Add it in the sidebar or via *Settings → Secrets*.")
        st.stop()

    with st.status("Extracting keywords…", state="running"):
        resume_text = pdf_to_text(file)
        if not resume_text:
            st.error("Could not read text from your PDF. Please try another file.")
            st.stop()
        keywords = extract_keywords(resume_text, top_n=12)
        st.write("**Extracted keywords:**", ", ".join(keywords) if keywords else "(none)")

    query = build_query(keywords) if keywords else ""
    st.write("**Using query:**", query or "(fallback to job title keywords)")

    with st.status("Fetching jobs…", state="running"):
        jobs = fetch_jobs(
            query=query or "software engineer OR backend engineer OR data engineer",
            days=days,
            pages=pages,
            country=country,
            remote_only=remote_only,
            rapid_key=api_key,
            rapid_host=host,
        )
        st.write(f"Fetched **{len(jobs)}** jobs.")

    if not jobs:
        st.warning("No jobs found. Try different days/country or broaden the resume keywords.")
        st.stop()

    with st.status("Ranking by resume match…", state="running"):
        ranked = rank_jobs(resume_text, jobs, top_k=30)

    st.success(f"Top {len(ranked)} matches")
    for i, row in ranked.iterrows():
        st.markdown(f"### {i+1}. {row['title']}  \n**Company:** {row['company']}  \n**Location:** {row['location'] or '—'}")
        if row["posted_utc"]:
            st.caption(f"Posted: {row['posted_utc']}")
        st.progress(float(max(0.0, min(1.0, row['score']))))
        c1, c2 = st.columns([0.18, 0.82])
        with c1:
            if row["apply_url"]:
                st.link_button("Apply", row["apply_url"])
        with c2:
            with st.expander("Description"):
                st.write(row["description"][:2000])

st.divider()
with st.expander("Tips & Next Steps"):
    st.markdown(
        "- If results are sparse, increase *days* or *pages* in the sidebar.\n"
        "- Add your **RapidAPI key** in *⋯ → Settings → Secrets* as `RAPIDAPI_KEY`.\n"
        "- This demo ranks results with **TF-IDF + cosine** (lightweight & fast)."
    )
