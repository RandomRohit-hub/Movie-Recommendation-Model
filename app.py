import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Configuration ---
st.set_page_config(page_title="Movie Recommender 🎬", page_icon="🎥", layout="centered")

# --- Title and Description ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Find similar movies based on content & description</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("movie.csv.csv")

movies_data = load_data()

# --- Preprocessing ---
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = (
    movies_data['genres'] + ' ' +
    movies_data['keywords'] + ' ' +
    movies_data['tagline'] + ' ' +
    movies_data['cast'] + ' ' +
    movies_data['director']
)

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

# --- Input UI ---
st.markdown("#### 🔍 Search for a movie:")
movie_name = st.text_input("Enter movie title", placeholder="e.g. Inception")

# --- Recommendation Logic ---
if movie_name:
    movie_list = movies_data['title'].tolist()
    close_matches = difflib.get_close_matches(movie_name, movie_list)

    if close_matches:
        close_match = close_matches[0]
        index_of_movie = movies_data[movies_data.title == close_match].index[0]
        similarity_scores = list(enumerate(similarity[index_of_movie]))
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:6]

        st.markdown(f"### 🎯 Recommendations for **{close_match}**")
        for i, _ in sorted_movies:
            st.markdown(f"✅ {movies_data.iloc[i]['title']}")
    else:
        st.error("❌ No close match found. Try a different title.")

# --- Footer ---
st.markdown("---")
st.markdown("<small>🚀 Built with Streamlit | Content-based Recommendation</small>", unsafe_allow_html=True)
