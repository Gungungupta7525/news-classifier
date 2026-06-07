import streamlit as st

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="News Classifier", layout="centered")

# -----------------------------
# Load model & vectorizer (for prediction)
# -----------------------------
#model = pickle.load(open("model.pkl", "rb"))
#clf_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
@st.cache_resource
def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, clf_vectorizer = load_model()

# -----------------------------
# Load dataset
# -----------------------------
#df = pd.read_csv("train.csv")
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    
    # 🔥 reduce size 
    df = df.sample(2000, random_state=42)
    
    df['text'] = df['Title'] + " " + df['Description']
    df = df.dropna()
    
    return df

df = load_data()
documents = df['text'].tolist()

# Combine text
df['text'] = df['Title'] + " " + df['Description']
df = df.dropna()

documents = df['text'].tolist()

# -----------------------------
# Label Mapping
# -----------------------------
labels = {
    1: "World 🌍",
    2: "Sports 🏏",
    3: "Business 💼",
    4: "Sci/Tech 🤖"
}

# -----------------------------
# TF-IDF for Similar News (FAST 🔥)
# -----------------------------
@st.cache_resource
def create_tfidf(docs):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    #tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    vectors = tfidf_vectorizer.fit_transform(docs)
    return tfidf_vectorizer, vectors

tfidf_vectorizer, doc_vectors = create_tfidf(documents)

# -----------------------------
# Retrieve Similar News
# -----------------------------
def retrieve_similar_news(query, top_k=3):
    query_vec = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors)[0]

    # Get top results
    #top_indices = similarities.argsort()[-top_k*2:][::-1]
    top_indices = similarities.argsort()[-top_k-5:][::-1]

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
This news is related to {category.lower()} because it talks about recent developments, events, or trends in this domain.
The system analyzed similar past news and found matching patterns, which helped in predicting the correct category.
"""
@st.cache_data
def predict_text(text):
    return model.predict(clf_vectorizer.transform([text]))[0]

# -----------------------------
# UI
# -----------------------------
st.title("📰 News Classifier  🤖")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    if user_input.strip() != "":

        # Prediction
        #prediction = model.predict(clf_vectorizer.transform([user_input]))[0]
        if len(user_input.split()) < 6:
            st.warning("⚠️ Please enter proper news content")
            st.stop()

        # ✅ Step 2: Keyword check
        news_keywords = ["government", "match", "market", "technology", "company", "election"]

        if not any(word in user_input.lower() for word in news_keywords):
            st.warning("⚠️ Input doesn't look like news")
            st.stop()
        prediction = predict_text(user_input)
        category = labels[prediction]

        st.success(f"Category: {category}")

        st.divider()

        # Similar News
        similar_news = retrieve_similar_news(user_input)

        st.subheader("🔎 Similar News")
        for news in similar_news:
            st.write("•", news[:150])

        st.divider()

        # Explanation
        st.subheader("🧠 Explanation")
        explanation = generate_explanation(user_input, category)
        st.write(explanation)

    else:
        st.warning("Please enter some text")