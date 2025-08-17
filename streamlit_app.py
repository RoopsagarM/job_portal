# streamlit_app.py
import re
import requests
import streamlit as st
from PyPDF2 import PdfReader

# ---- Optional KeyBERT, with lightweight fallback (TF-IDF) ----
try:
    from keybert import KeyBERT
    _kw = KeyBERT()

    def extract_keywords_from_text(text: str, top_n: int = 10):
        return [k[0] for k in _kw.extract_keywords(text, top_n=top_n)]
except Exception:
    # Fallback: tiny TF-IDF extractor (no heavy deps)
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    def extract_keywords_from_text(text: str, top_n: int = 10):
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
                seen.add(t)
                out.append(t)
            if len(out) >= top_n:
                break
        return out

# --------------------------
# Parse years of experience
# --------------------------
EXP_PATTERNS = [
    r"(\d{1,2})\s*\+?\s*(?:years|yrs)\s+of\s+experience",
    r"(\d{1,2})\s*\+?\s*(?:years|yrs)\b",             # e.g., "5 years", "3+ yrs"
    r"(\d{1,2})\s*-\s*(\d{1,2})\s*(?:years|yrs)\b",   # "3-5 years"
]

def parse_experience_years(text: str) -> float:
    text = text.lower()
    years = []

    # ranges first (take the max of the range)
    for m in re.finditer(r"(\d{1,2})\s*-\s*(\d{1,2})\s*(?:years|yrs)\b", text):
        a, b = int(m.group(1)), int(m.group(2))
        years.append(max(a, b))

    # explicit "X years of experience"
    for m in re.finditer(r"(\d{1,2})\s*\+?\s*(?:years|yrs)\s+of\s+experience", text):
        years.append(int(m.group(1)))

    # generic "X years/yrs"
    for m in re.finditer(r"(?<!-)\b(\d{1,2})\s*\+?\s*(?:years|yrs)\b", text):
        years.append(int(m.group(1)))

    # CVs often list roles like "Senior Software Engineer (2018–2023)"
    # You can enhance with date math later if needed.

    return float(max(years)) if years else 0.0

def seniority_from_years(years: float) -> str:
    if years <= 1.5:
        return "entry"
    if years <= 5:
        return "mid"
    return "senior"

# --------------------------
# Query builder (keeps your logic)
# --------------------------
def build_smart_query(keywords):
    text = " ".join(keywords).lower()
    query = "software engineer OR developer"  # default fallback

    if any(k in text for k in ["data", "sql", "python", "pandas", "analysis", "ml", "ai"]):
        query = "data analyst OR data engineer OR machine learning engineer OR data scientist"
    elif any(k in text for k in ["aws", "cloud", "distributed", "redshift", "snowflake", "devops"]):
        query = "cloud engineer OR backend engineer OR devops engineer OR software engineer"
    elif any(k in text for k in ["react", "javascript", "frontend", "ui", "web"]):
        query = "frontend developer OR full stack engineer OR web developer"

    return query

# --------------------------
# Fetch jobs (RapidAPI → JSearch)
# --------------------------
def fetch_jobs(query: str, days: int = 7, max_jobs: int = 40, country="us", remote_only=True):
    url = "https://jsearch.p.rapidapi.com/search"
    params = {
        "query": query,
        "page": 1,
        "num_pages": 3,
        "date_posted": "month" if days >= 30 else ("week" if days >= 7 else ("3days" if days >= 3 else "today")),
        "country": country,
        "remote_only": "true" if remote_only else "false",
    }

    # Prefer Secrets → env → fallback (your current hardcoded key)
    api_key = st.secrets.get("RAPIDAPI_KEY", "") or \
              os.getenv("RAPIDAPI_KEY", "") or \
              "f3bd6b71a8mshf338bb20bddde4fp1f4e9ajsnf9be665c5554"  # replace in prod
    headers = {
        "X-RapidAPI-Key": "f3bd6b71a8mshf338bb20bddde4fp1f4e9ajsnf9be665c5554",
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        st.error(f"Error fetching jobs: {r.text[:300]}")
        return []
    return (r.json() or {}).get("data", [])[:max_jobs]

# --------------------------
# Filter jobs by seniority
# --------------------------
SENIOR_WORDS = ["senior", "lead", "principal", "staff", "architect", "manager", "head"]
ENTRY_WORDS  = ["junior", "entry", "intern", "new grad", "graduate", "assoc", "associate"]

def title_matches_level(title: str, level: str) -> bool:
    t = (title or "").lower()
    if not t:
        return True

    if level == "senior":
        return any(w in t for w in SENIOR_WORDS)
    if level == "entry":
        # reject obvious senior titles
        if any(w in t for w in SENIOR_WORDS):
            return False
        return any(w in t for w in ENTRY_WORDS) or True  # most titles OK for entry if not marked senior
    # mid:
    if any(w in t for w in SENIOR_WORDS) or any(w in t for w in ENTRY_WORDS):
        return False
    return True

def filter_jobs_by_level(jobs, level: str):
    return [j for j in jobs if title_matches_level(j.get("job_title", ""), level)]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Resume → Fresh Job Finder", page_icon="🔎", layout="centered")
st.title("🔍 Resume → Fresh Job Finder")
st.write("Upload your resume → auto keywords → detect experience → fetch recent jobs → filtered by level")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Read PDF -> text
    resume_text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        resume_text += page.extract_text() or ""

    # Detect experience & level
    years = parse_experience_years(resume_text)
    level = seniority_from_years(years)
    st.info(f"**Detected experience:** ~{years:.0f} years  •  **Seniority:** {level.title()}")

    # Extract keywords
    with st.spinner("Extracting keywords…"):
        keywords = extract_keywords_from_text(resume_text, top_n=10)
        st.write("**Keywords:**", ", ".join(keywords) if keywords else "—")

    # Build query & fetch jobs
    query = build_smart_query(keywords)
    st.write("**Query:**", query)

    with st.spinner("Fetching jobs…"):
        jobs = fetch_jobs(query, days=30, country="us", max_jobs=60, remote_only=True)

    # Filter by detected level
    jobs_level = filter_jobs_by_level(jobs, level)

    # If filter got too strict, fall back to unfiltered
    results = jobs_level if jobs_level else jobs

    if not results:
        st.warning("No jobs found. Try widening search (country/days) or upload a different resume.")
    else:
        st.success(f"Showing {len(results)} job(s) for **{level.title()}** profiles")
        for job in results:
            title = job.get("job_title") or "Untitled role"
            company = job.get("employer_name") or "—"
            city = job.get("job_city") or job.get("job_location") or ""
            country = job.get("job_country") or ""
            posted = job.get("job_posted_at_datetime_utc") or job.get("job_posted_at") or ""
            desc = (job.get("job_description") or job.get("description") or "")[:400].strip()
            link = job.get("job_apply_link") or job.get("job_url")

            st.subheader(title)
            st.write(f"**Company:** {company}")
            st.write(f"**Location:** {', '.join([p for p in [city, country] if p]) or '—'}")
            if posted:
                st.caption(f"Posted: {posted}")
            if desc:
                st.write(desc + ("…" if len(desc) == 400 else ""))
            if link:
                st.markdown(f"[Apply here]({link})")
            st.markdown("---")
