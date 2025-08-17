import streamlit as st
import requests
from keybert import KeyBERT
from PyPDF2 import PdfReader

# --------------------------
# Extract keywords from resume
# --------------------------
kw_model = KeyBERT()

def extract_keywords_from_resume(text, top_n=10):
    keywords = [k[0] for k in kw_model.extract_keywords(text, top_n=top_n)]
    return keywords

# --------------------------
# Build smart job query
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
# Fetch jobs from RapidAPI
# --------------------------
def fetch_jobs(query: str, days: int = 7, max_jobs: int = 20, country="us"):
    url = "https://jsearch.p.rapidapi.com/search"
    params = {
        "query": query,
        "page": 1,
        "num_pages": 3,             # fetch more pages
        "date_posted": "month",     # broaden to last 30 days
        "country": country,
        "remote_only": "true"       # only remote jobs
    }
    headers = {
        "X-RapidAPI-Key": "f3bd6b71a8mshf338bb20bddde4fp1f4e9ajsnf9be665c5554",   # 🔑 Replace with your key
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }

    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        st.error(f"Error fetching jobs: {r.text}")
        return []

    return r.json().get("data", [])[:max_jobs]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Resume → Fresh Job Finder", layout="centered")
st.title("🔍 Resume → Fresh Job Finder")
st.write("Upload your resume → auto keywords → fetch recent jobs → ranked by fit")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text() or ""

    # Extract keywords
    with st.spinner("Extracting keywords..."):
        keywords = extract_keywords_from_resume(resume_text)
        st.write("**Extracted keywords:**", ", ".join(keywords))

    # Build query
    query = build_smart_query(keywords)
    st.write("**Using query:**", query)

    # Fetch jobs
    with st.spinner("Fetching jobs..."):
        jobs = fetch_jobs(query, days=30, country="us")

    if not jobs:
        st.warning("⚠️ No jobs found. Try again later or widen search.")
    else:
        st.success(f"Found {len(jobs)} jobs")
        for job in jobs:
            st.subheader(job.get("job_title"))
            st.write(f"**Company:** {job.get('employer_name')}")
            st.write(f"**Location:** {job.get('job_city')}, {job.get('job_country')}")
            st.write(f"**Posted:** {job.get('job_posted_at_datetime_utc')}")
            st.write(job.get("job_description")[:250] + "...")
            if job.get("job_apply_link"):
                st.markdown(f"[Apply Here]({job['job_apply_link']})")
            st.markdown("---")