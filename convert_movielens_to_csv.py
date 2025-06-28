import pandas as pd

# ----- Convert u.data to ratings.csv -----
ratings = pd.read_csv("ml-100k/u.data", sep="\t", header=None)
ratings.columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings.to_csv("ratings.csv", index=False)

# ----- Convert u.item to movies.csv -----
movie_cols = [
    "movieId", "title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies = pd.read_csv("ml-100k/u.item", sep="|", encoding='latin-1', header=None, names=movie_cols)

# Add 'genres' column by combining genre flags
genre_columns = movie_cols[5:]  # All genre flags
def extract_genres(row):
    return '|'.join([genre for genre in genre_columns if row[genre] == 1])

movies['genres'] = movies.apply(extract_genres, axis=1)

# Save as movies.csv
movies.to_csv("movies.csv", index=False)
