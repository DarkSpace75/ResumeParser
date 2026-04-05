import re
import PyPDF2
import numpy as np
import joblib
from gensim.models.doc2vec import Doc2Vec
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained models
model = Doc2Vec.load("cv_job_matching.model")
svm_classifier = joblib.load("svm_job_classifier.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Extract text from PDF
def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return preprocess_text(text)

# Match resume with job descriptions
def match_resume_with_jobs(resume_text, df):
    resume_vector = model.infer_vector(resume_text.split())

    best_match = None
    best_similarity = 0

    for _, row in df.iterrows():
        job_desc = preprocess_text(row["Additional Information"])
        job_vector = model.infer_vector(job_desc.split())

        similarity = 100 * np.dot(resume_vector, job_vector) / (norm(resume_vector) * norm(job_vector))

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = row["Business Title"]

    return best_match, round(best_similarity, 2)

# Predict job category using SVM
def predict_job_category(resume_text):
    processed_text = preprocess_text(resume_text)
    prediction = svm_classifier.predict([processed_text])
    predicted_category = label_encoder.inverse_transform(prediction)[0]
    return predicted_category
