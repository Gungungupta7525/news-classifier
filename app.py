import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="News Classifier", layout="centered")

# -----------------------------
# Load model & vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, clf_vectorizer = load_model()

# -----------------------------
# Load dataset (LIGHT VERSION)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    # 🔥 Reduce size (important for Render free)
    df = df.sample(2000, random_state=42)

    df['text'] = df['Title'] + " " + df['Description']
    df = df.dropna()

    return df

df = load_data()
documents = df['text'].tolist()

# -----------------------------
# Label Mapping
# -----------------------------
labels = {
    0: "World 🌍",
    1: "Sports 🏏",
    2: "Business 💼",
    3: "Sci/Tech 🤖"
}

# -----------------------------
# TF-IDF for Similar News
# -----------------------------
@st.cache_resource
def create_tfidf(docs):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectors = tfidf_vectorizer.fit_transform(docs)
    return tfidf_vectorizer, vectors

tfidf_vectorizer, doc_vectors = create_tfidf(documents)

# -----------------------------
# Retrieve Similar News
# -----------------------------
def retrieve_similar_news(query, top_k=3):
    query_vec = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors)[0]

    top_indices = similarities.argsort()[-(top_k+5):][::-1]

    results = []
    seen = set()

    for idx in top_indices:
        text = df.iloc[idx]['text']
        if text not in seen:
            results.append(text)
            seen.add(text)
        if len(results) == top_k:
            break

    return results

# -----------------------------
# Explanation
# -----------------------------
def generate_explanation(query, category):
    return f"""
This news belongs to **{category}** category.

👉 News:
{query}

👉 Explanation:
The system analyzed patterns from previously seen news data and classified this input into the {category} category based on similarity and learned features.
"""

# -----------------------------
# UI
# -----------------------------
st.title("📰 News Classifier with Smart AI 🤖")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    if user_input.strip() != "":

        # -----------------------------
        # ML-based validation (NO keywords 🔥)
        # -----------------------------
        probs = model.predict_proba(clf_vectorizer.transform([user_input]))[0]

        confidence = max(probs)
        prediction = probs.argmax()

        if confidence < 0.4:
            st.warning("⚠️ Input doesn't look like valid news content")
            st.stop()

        category = labels[prediction]

        # -----------------------------
        # Output
        # -----------------------------
        st.success(f"Category: {category}")
        #st.write(f"Confidence: {confidence:.2f}")

        st.divider()

        # -----------------------------
        # Similar News
        # -----------------------------
        st.subheader("🔎 Similar News")
        similar_news = retrieve_similar_news(user_input)

        for news in similar_news:
            st.write("•", news[:150])

        st.divider()

        # -----------------------------
        # Explanation
        # -----------------------------
        st.subheader("🧠 Explanation")
        explanation = generate_explanation(user_input, category)
        st.write(explanation)

    else:
        st.warning("Please enter some text")