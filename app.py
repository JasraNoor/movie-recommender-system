# app.py
from flask import Flask, render_template, request, jsonify
from recommendation_model import MovieRecommender
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Load the trained model
try:
    recommender = MovieRecommender.load_model('movie_recommender.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model not found. Please train the model first.")
    recommender = None

@app.route('/')
def home():
    """Home page with recommendation options"""
    if recommender is None:
        return "Model not loaded. Please train the model first."
    
    # Get some popular movies for display
    popular_movies = recommender.get_popular_movies(20)
    movie_details = recommender.get_movie_details(popular_movies)
    
    return render_template('index.html', popular_movies=movie_details)

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """Get recommendations based on user input"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'})
    
    try:
        recommendation_type = request.form.get('type')
        
        if recommendation_type == 'collaborative':
            user_id = int(request.form.get('user_id'))
            movie_ids = recommender.get_collaborative_recommendations(user_id, 10)
            
        elif recommendation_type == 'content':
            movie_id = int(request.form.get('movie_id'))
            movie_ids = recommender.get_content_recommendations(movie_id, 10)
            
        else:
            movie_ids = recommender.get_popular_movies(10)
        
        # Get movie details
        recommendations = recommender.get_movie_details(movie_ids)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'type': recommendation_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/search_movies')
def search_movies():
    """Search movies by title"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'})
    
    query = request.args.get('q', '').lower()
    
    if len(query) < 2:
        return jsonify({'movies': []})
    
    # Search in movie titles
    matching_movies = recommender.movies[
        recommender.movies['title'].str.lower().str.contains(query, na=False)
    ].head(10)
    
    movies = matching_movies[['movieId', 'title', 'genres']].to_dict('records')
    
    return jsonify({'movies': movies})

@app.route('/movie_stats')
def movie_stats():
    """Get basic statistics about the dataset"""
    if recommender is None:
        return jsonify({'error': 'Model not loaded'})
    
    stats = {
        'total_movies': len(recommender.movies),
        'total_ratings': len(recommender.ratings),
        'total_users': recommender.ratings['userId'].nunique(),
        'avg_rating': round(recommender.ratings['rating'].mean(), 2),
        'top_genres': recommender.movies['genres'].str.split('|').explode().value_counts().head(5).to_dict()
    }
    
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)