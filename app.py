import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import spacy

st.markdown("""
    <style>
    .main {
        background-color: #f4f6f9;
        padding: 2rem;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        color: #2c3e50;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.07);
        margin: 1rem 0;
        text-align: center;
    }
    .metric-title {
        font-size: 20px;
        font-weight: 600;
        color: #34495e;
    }
    .metric-value {
        font-size: 26px;
        color: #16a085;
    }
    </style>
""", unsafe_allow_html=True)


# Load SpaCy model (only once)
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_md")

nlp = load_spacy_model()

# --- Utility Functions ---
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

def extract_keywords(text):
    return set(re.findall(r'\b\w{3,}\b', text.lower()))

def tfidf_score(text1, text2):
    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform([text1, text2])
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100

def semantic_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2) * 100

# --- UI Setup ---
st.set_page_config(page_title="Resume Analyzer", layout="centered")
st.markdown('<div class="title">📄 Resume Analyzer</div>', unsafe_allow_html=True)

resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the Job Description Here")

# --- Main Logic ---
if resume_file and job_description:
    resume_text = extract_text_from_pdf(resume_file)

    # Compute Metrics
    tfidf = tfidf_score(resume_text, job_description)
    semantic = semantic_similarity(resume_text, job_description)

    resume_keywords = extract_keywords(resume_text)
    job_keywords = extract_keywords(job_description)
    matched = len(resume_keywords & job_keywords)
    keyword_percent = (matched / len(job_keywords)) * 100 if job_keywords else 0

    combined = 0.6 * semantic + 0.2 * tfidf + 0.2 * keyword_percent

    # --- Display Results ---
    st.markdown('<div class="metric-card"><div class="metric-title">🔑 Keyword Match</div><div class="metric-value">{:.2f}%</div></div>'.format(keyword_percent), unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><div class="metric-title">🧠 Semantic Similarity</div><div class="metric-value">{:.2f}%</div></div>'.format(semantic), unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><div class="metric-title">🧮 TF-IDF Similarity</div><div class="metric-value">{:.2f}%</div></div>'.format(tfidf), unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><div class="metric-title">📊 Combined Score</div><div class="metric-value">{:.2f}%</div></div>'.format(combined), unsafe_allow_html=True)

    st.progress(combined / 100)

    if combined > 75:
        st.success("✅ Excellent Match - Your resume fits this role well.")
    elif combined > 50:
        st.warning("🟡 Partial Match - Consider tailoring your resume further.")
    else:
        st.error("❌ Poor Match - Improve alignment with job keywords and context.")
