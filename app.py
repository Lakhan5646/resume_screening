import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import PyPDF2

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ==============================
# 🔹 CLEAN TEXT
# ==============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]

    return " ".join(words)

# ==============================
# 🔹 LOAD MODEL
# ==============================
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ==============================
# 🔹 UI
# ==============================
st.title("📄 Resume Screening System")
st.write("Upload or paste resume to predict job category")

# ==============================
# 🔹 TEXT INPUT
# ==============================
text_input = st.text_area("Paste Resume Text")

# ==============================
# 🔹 FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

resume_text = ""

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        resume_text += page.extract_text()

# ==============================
# 🔹 PREDICTION
# ==============================
if st.button("Predict"):

    if text_input:
        resume_text = text_input

    if resume_text:
        cleaned = clean_text(resume_text)
        vec = vectorizer.transform([cleaned])
        result = model.predict(vec)

        st.success(f"Predicted Category: {result[0]}")

    else:
        st.warning("Please provide resume text or upload PDF")