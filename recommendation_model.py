# recommendation_model.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class MovieRecommender:
    def __init__(self):
        self.ratings = None
        self.movies = None
        self.user_movie_matrix = None
        self.movie_similarity = None
        self.content_similarity = None
        
    def load_data(self, ratings_path, movies_path):
        """Load ratings and movies data"""
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)
        print(f"Loaded {len(self.ratings)} ratings and {len(self.movies)} movies")
        
    def prepare_collaborative_filtering(self):
        """Prepare user-item matrix for collaborative filtering"""
        # Create user-movie matrix
        self.user_movie_matrix = self.ratings.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Calculate movie-movie similarity
        movie_similarity = cosine_similarity(self.user_movie_matrix.T)
        self.movie_similarity = pd.DataFrame(
            movie_similarity,
            index=self.user_movie_matrix.columns,
            columns=self.user_movie_matrix.columns
        )
        
    def prepare_content_filtering(self):
        """Prepare content-based filtering using movie genres"""
        # Combine genres for TF-IDF
        self.movies['genres_clean'] = self.movies['genres'].str.replace('|', ' ')
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies['genres_clean'])
        
        # Calculate content similarity
        self.content_similarity = cosine_similarity(tfidf_matrix)
        
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using collaborative filtering"""
        if user_id not in self.user_movie_matrix.index:
            return self.get_popular_movies(n_recommendations)
            
        user_ratings = self.user_movie_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        
        # Calculate weighted ratings for unrated movies
        recommendations = {}
        
        for movie in self.user_movie_matrix.columns:
            if movie not in rated_movies:
                # Get similar movies that user has rated
                similar_movies = self.movie_similarity[movie]
                similar_rated = similar_movies[rated_movies]
                
                if len(similar_rated) > 0:
                    # Weighted average of similar movies
                    numerator = (similar_rated * user_ratings[rated_movies]).sum()
                    denominator = similar_rated.sum()
                    
                    if denominator > 0:
                        recommendations[movie] = numerator / denominator
        
        # Sort and return top recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [movie_id for movie_id, score in sorted_recs[:n_recommendations]]
    
    def get_content_recommendations(self, movie_id, n_recommendations=10):
        """Get recommendations using content-based filtering"""
        if movie_id not in self.movies['movieId'].values:
            return []
            
        # Get movie index
        movie_idx = self.movies[self.movies['movieId'] == movie_id].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.content_similarity[movie_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar movies (excluding the movie itself)
        similar_movies = sim_scores[1:n_recommendations+1]
        movie_indices = [i[0] for i in similar_movies]
        
        return self.movies.iloc[movie_indices]['movieId'].tolist()
    
    def get_popular_movies(self, n_recommendations=10):
        """Get popular movies for new users"""
        popular = self.ratings.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).round(2)
        
        popular.columns = ['avg_rating', 'rating_count']
        popular = popular[popular['rating_count'] >= 50]  # Min 50 ratings
        popular = popular.sort_values(['avg_rating', 'rating_count'], ascending=False)
        
        return popular.head(n_recommendations).index.tolist()
    
    def get_movie_details(self, movie_ids):
        """Get movie details for given IDs"""
        return self.movies[self.movies['movieId'].isin(movie_ids)][
            ['movieId', 'title', 'genres']
        ].to_dict('records')
    
    def save_model(self, filepath):
        """Save the trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Training script
if __name__ == "__main__":
    recommender = MovieRecommender()
    
    # Load and prepare data
    recommender.load_data('ratings.csv', 'movies.csv')
    recommender.prepare_collaborative_filtering()
    recommender.prepare_content_filtering()
    
    # Save the model
    recommender.save_model('movie_recommender.pkl')
    print("Model saved successfully!")