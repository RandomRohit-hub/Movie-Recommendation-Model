# 🎬 Movie Recommender System

A simple content-based movie recommendation system built using **Streamlit** and **Scikit-learn**. Enter the name of a movie, and get a list of similar movies based on genres, keywords, tagline, cast, and director.

---

## 🚀 Demo
https://movie-recommendation-model-7rlxdnkigpwovswcgyzmfz.streamlit.app/<!-- Optional: Add a gif or screenshot -->

---

## 🔧 Features

- Content-based filtering using TF-IDF and cosine similarity
- Streamlit web interface with clean UI
- Auto-correction for movie title inputs
- Recommends top 5 similar movies

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit

---

## 📁 Dataset

The dataset (`movie.csv.csv`) contains metadata about movies, including:

- `title`
- `genres`
- `keywords`
- `tagline`
- `cast`
- `director`

> 📌 Make sure the CSV file is in the same folder as `app.py`.

---

## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
