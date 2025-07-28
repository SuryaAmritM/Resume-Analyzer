import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from functools import lru_cache

@lru_cache(maxsize=128)
def get_sentence_embeddings(text):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) > 4]
    embeddings = st_model.encode(sentences)
    return embeddings, sentences

def clean_text(text):
    # Remove multiple spaces, newlines, tabs, etc.
    text = re.sub(r'\s+', ' ', text.strip())
    return text.lower()

def analyze_sentences_and_suggest(resume_text, jd_text):
    resume_sent_embeds, resume_sents = get_sentence_embeddings(resume_text)
    jd_sent_embeds, jd_sents = get_sentence_embeddings(jd_text)

    # Document embeddings
    resume_doc_embed = st_model.encode(resume_text, convert_to_tensor=True)
    jd_doc_embed = st_model.encode([jd_text])[0]

    # Sentence-level similarity
    similarities = cosine_similarity(jd_sent_embeds, resume_sent_embeds)
    best_matches = similarities.max(axis=1)
    average_sent_score = np.mean(best_matches)

    # Document-level similarity
    doc_score = cosine_similarity([jd_doc_embed], [resume_doc_embed])[0][0]

    # Combined score
    final_score = 0.6 * doc_score + 0.4 * average_sent_score

    # Suggestions
    worst_index = best_matches.argmin()
    missing_info = jd_sents[worst_index]
    top_indices = best_matches.argsort()[-3:][::-1]
    top_matches = [
        (jd_sents[i], resume_sents[similarities[i].argmax()], best_matches[i])
        for i in top_indices
    ]

    suggestions = f"Consider covering this in your resume: \"{missing_info}\" — it's not well matched."

    return {
        "final_score": round(final_score * 100, 2),
        "doc_score": round(doc_score * 100, 2),
        "sentence_score": round(average_sent_score * 100, 2),
        "top_matches": top_matches,
        "suggestion": suggestions
    }

# Load SpaCy model with vectors
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_md")

@st.cache_resource
def load_st_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

nlp = load_spacy_model()
st_model = load_st_model()

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

    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # unigrams, bigrams, trigrams
    tfidf_matrix = vectorizer.fit_transform([resume_lem, jd_lem])
    score = (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
    return score * 100


# -----------------------
# Semantic Score
# -----------------------
def get_semantic_score(resume, jd):
    embeddings = st_model.encode([resume, jd])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity * 100

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

    # Analyze top/bottom semantic matches and give suggestions
    result = analyze_sentences_and_suggest(resume_text, jd_text)
    top_pairs = result["top_matches"]
    missing_info = result["suggestion"].split(":")[1].split("—")[0].strip(' "')
    suggestions = result["suggestion"]

    with st.expander("🧠 Semantic Insights"):
        st.subheader("🔍 Most Aligned Sentences")
        for jd_sent, res_sent, score in top_pairs:
            st.markdown(f"**JD:** {jd_sent}")
            st.markdown(f"**Resume:** {res_sent}")
            st.markdown(f"**Similarity:** {score:.2f}%")
            st.markdown("---")

        st.subheader("❗ Possibly Missing or Weakly Covered JD Content")
        st.warning(f"`{missing_info}`")

        st.subheader("💡 Suggestion to Improve Resume")
        st.info(suggestions)

    st.markdown("---")
    if combined_score > 75:
        st.success("✅ Excellent Match! Your resume is well-aligned.")
    elif combined_score > 50:
        st.info("⚠️ Decent match. Consider improving keywords or tailoring content.")
    else:
        st.warning("❌ Low match. Customize your resume more for this job.")

else:
    st.info("Upload a resume and paste a job description to get started.")
