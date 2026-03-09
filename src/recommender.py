from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import DBSCAN
import pandas as pd
import os

def load_recommender(csv_path):
    """
    Loads data and prepares the recommendation system components.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing dataset at {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['Book_Name_x'] = df['Book_Name_x'].str.lower()
    
    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Book_Name_x'])
    
    # Calculate Similarity
    cosine_sim = linear_kernel(X, X)
    
    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    cluster_labels = dbscan.fit_predict(X)
    df['cluster'] = cluster_labels
    
    return df, cosine_sim

def recommend(book_title, df, cosine_sim, top_n=5):
    """
    Generates book recommendations based on title similarity.
    """
    book_title = book_title.lower()
    if book_title not in df['Book_Name_x'].values:
        return None
    
    idx = df[df['Book_Name_x'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:top_n+1]]

    # Ensure metadata columns exist
    cols = ['Book_Name_x', 'Author_x', 'Rating_x', 'Description']
    available_cols = [c for c in cols if c in df.columns]
    
    return df.iloc[sim_indices][available_cols]
