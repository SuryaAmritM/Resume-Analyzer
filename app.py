import streamlit as st
import PyPDF2
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model
nlp = spacy.load("en_core_web_md")

st.set_page_config(page_title="Smart Resume Matcher", layout="centered")
st.title("📄 AI Resume Matcher")
st.write("Upload your resume and job description to see how well they align.")

# --- Helper Functions ---
def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return list(set(words))

def get_tfidf_score(resume, job):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume, job])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return score

def get_semantic_score(resume, job):
    doc1 = nlp(resume)
    doc2 = nlp(job)
    return doc1.similarity(doc2) * 100

# --- Upload Section ---
resume_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])
job_desc = st.text_area("Paste the job description here")

if resume_file and job_desc:
    resume_text = extract_text(resume_file)

    # Scores
    tfidf_score = get_tfidf_score(resume_text, job_desc)
    semantic_score = get_semantic_score(resume_text, job_desc)
    match_percent = round((0.75 * semantic_score + 0.25 * tfidf_score), 2)

    # Keyword analysis
    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_desc)
    missing_keywords = set(job_keywords) - set(resume_keywords)
    keyword_match_percent = (len(set(resume_keywords) & set(job_keywords)) / len(set(job_keywords))) * 100 if job_keywords else 0

    # --- Results ---
    st.markdown("---")
    st.subheader("📊 Match Summary")
    st.metric("✅ Final Match Score", f"{match_percent:.2f}%")

    # Suggested improvements
    if missing_keywords:
        st.warning("🛠️ Add these keywords to improve alignment:")
        st.write(", ".join(sorted(list(missing_keywords)[:10])))
    else:
        st.success("✅ Your resume already includes the key terms!")

    # Optional deep dive
    with st.expander("🔍 Show Matching Details"):
        st.metric("🧠 Semantic Match", f"{semantic_score:.2f}%")
        st.metric("🔑 Keyword Match", f"{keyword_match_percent:.2f}%")
        st.metric("🧾 TF-IDF Score", f"{tfidf_score:.2f}%")

    st.markdown("---")
    st.info("💡 Tip: Tailor your resume keywords to the job description for higher ATS scores.")
