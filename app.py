import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load SpaCy model with vectors
nlp = spacy.load("en_core_web_md")

# -----------------------
# Text Extraction
# -----------------------
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# -----------------------
# TF-IDF Score
# -----------------------
def lemmatize_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def get_tfidf_score(resume, jd):
    resume_lem = lemmatize_text(resume)
    jd_lem = lemmatize_text(jd)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_lem, jd_lem])
    score = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
    return score * 100

# -----------------------
# Semantic Score
# -----------------------
def get_semantic_score(resume, jd):
    doc1 = nlp(resume)
    doc2 = nlp(jd)
    return doc1.similarity(doc2) * 100

# -----------------------
# Keyword Score
# -----------------------
def get_keyword_score(resume, jd):
    resume_words = set(re.findall(r'\w+', resume.lower()))
    jd_words = set(re.findall(r'\w+', jd.lower()))
    common = resume_words & jd_words
    if not jd_words:
        return 0
    return len(common) / len(jd_words) * 100

# -----------------------
# UI Setup
# -----------------------
st.set_page_config(page_title="Resume Analyzer", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            text-align: center;
            color: #003049;
        }
        .card {
            background-color: #ffffff;
            padding: 20px;
            margin: 10px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            text-align: center;
        }
        .score {
            font-size: 32px;
            font-weight: bold;
            color: #264653;
        }
        .label {
            font-size: 16px;
            color: #6c757d;
        }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1 class='title'>📄 Resume vs Job Description Analyzer</h1>", unsafe_allow_html=True)
st.markdown("Use AI to measure how well your resume matches the job description.")
st.markdown("---")

# Uploads
resume_file = st.file_uploader("📤 Upload your Resume (PDF)", type=["pdf"])
jd_text = st.text_area("💼 Paste the Job Description")

# Run scoring if both are provided
if resume_file and jd_text:
    resume_text = extract_text_from_pdf(resume_file)

    tfidf_score = get_tfidf_score(resume_text, jd_text)
    semantic_score = get_semantic_score(resume_text, jd_text)
    keyword_score = get_keyword_score(resume_text, jd_text)
    combined_score = (tfidf_score + semantic_score + keyword_score) / 3

    # Score Cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><div class='label'>📌 Keyword Match</div><div class='score'>{:.2f}%</div></div>".format(keyword_score), unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='label'>🧠 Semantic Score</div><div class='score'>{:.2f}%</div></div>".format(semantic_score), unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'><div class='label'>📊 TF-IDF Score</div><div class='score'>{:.2f}%</div></div>".format(tfidf_score), unsafe_allow_html=True)
        st.markdown("<div class='card'><div class='label'>🔥 Combined Score</div><div class='score'>{:.2f}%</div></div>".format(combined_score), unsafe_allow_html=True)

    st.markdown("---")
    if combined_score > 75:
        st.success("✅ Excellent Match! Your resume is well-aligned.")
    elif combined_score > 50:
        st.info("⚠️ Decent match. Consider improving keywords or tailoring content.")
    else:
        st.warning("❌ Low match. Customize your resume more for this job.")

else:
    st.info("Upload a resume and paste a job description to get started.")
