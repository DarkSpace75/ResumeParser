from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")

# Load dataset
df = pd.read_csv("nyc-jobs-1.csv")
df = df.dropna(subset=["Additional Information"])  # Remove empty job descriptions

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Prepare job descriptions
documents = [TaggedDocument(words=word_tokenize(preprocess_text(desc)), tags=[str(i)]) for i, desc in enumerate(df["Additional Information"])]

# Train model
model = Doc2Vec(documents, vector_size=100, window=5, min_count=1, workers=4, epochs=20)
model.save("cv_job_matching.model")

print("Model training complete. Saved as cv_job_matching.model")
