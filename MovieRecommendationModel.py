import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import matplotlib.pyplot as plt

# ---- Load CSV files ----
credits = pd.read_csv("tmdb_5000_credits.csv", encoding='ISO-8859-1')
movies = pd.read_csv("tmdb_5000_movies.csv", encoding='ISO-8859-1')

# ---- Merge and clean data ----
credits_renamed = credits.rename(columns={"movie_id": "id"})
movies_merged = movies.merge(credits_renamed, on="id")
movies_cleaned = movies_merged.drop(columns=['homepage', 'original_title', 'title_x'])

# ---- TF-IDF vectorization ----
tfv = TfidfVectorizer(
    min_df=3,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3),
    stop_words='english'
)
tfv_matrix = tfv.fit_transform(movies_cleaned['overview'].fillna(''))

# ---- Compute similarity ----
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

# ---- Map movie titles to indices ----
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['title_y']).drop_duplicates()

# ---- Recommendation function ----
def give_recommendations(title, sig=sig):
    if title not in indices:
        return f"Movie '{title}' not found in the dataset."
    
    idx = indices[title]
    sig_scores = list(enumerate(sig[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:11]  # Top 10 excluding the movie itself
    movie_indices = [i[0] for i in sig_scores]
    recommended_movies = movies_cleaned['title_y'].iloc[movie_indices]
    recommended_scores = [i[1] for i in sig_scores]
    
    return recommended_movies, recommended_scores

# ---- Get recommendations for a movie ----
movie_name = 'Tangled'
recommendations, scores = give_recommendations(movie_name)

# ---- Plot recommendations ----
# Use a valid built-in style
plt.style.use('fivethirtyeight')  # modern, clean, built-in Matplotlib style
plt.figure(figsize=(10, 6))
plt.barh(recommendations[::-1], scores[::-1], color="#1f77b4")  # highest score on top
plt.xlabel("Similarity Score", fontsize=12)
plt.ylabel("Movie Name", fontsize=12)
plt.title(f"Top 10 Recommendations for '{movie_name}'", fontsize=16)
plt.xticks(fontsize=10)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.show()
