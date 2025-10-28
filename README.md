A content-based movie recommendation system built in Python that suggests movies similar to a given title using TF-IDF vectorization and cosine similarity. The recommendations are visualized in a modern, clean Matplotlib chart.

Features

Provides top 10 movie recommendations based on movie overviews.

Uses TF-IDF to vectorize movie descriptions and compute similarity.

Clean horizontal bar chart visualization of similarity scores.

Handles missing data and invalid movie titles gracefully.

Installation

Clone this repository:

git clone https://github.com/your-username/movie-recommendation.git


Install required packages:

pip install pandas scikit-learn matplotlib

Usage

Place the CSV files (tmdb_5000_credits.csv and tmdb_5000_movies.csv) in the project directory.

Run the script:

python Movie_Recommendation.py


Input a movie title to get top 10 similar movies along with a clean bar chart of similarity scores.

Example:

recommendations, scores = give_recommendations('Tangled')

Example Output

Recommendations for 'Tangled' will display a horizontal bar chart showing the similarity scores of the top 10 recommended movies.

Dataset

Movie datasets used are from TMDB 5000
.

Ensure CSV files are UTF-8 or ISO-8859-1 encoded to avoid errors.

License

This project is open-source and free to use under the MIT License.
