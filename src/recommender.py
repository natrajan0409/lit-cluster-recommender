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
    
    # Create a combined feature for vectorization
    # Weight genres and titles more heavily by repeating them
    df['metadata'] = (
        (df['Book_Name_x'].fillna('') + ' ') * 2 + 
        df['Description'].fillna('') + ' ' + 
        (df['Ranks and Genre'].fillna('') + ' ') * 3
    ).str.lower()
    
    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['metadata'])
    
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
    cols = ['Book_Name_x', 'Author_x', 'Rating_x', 'Description', 'Ranks and Genre']
    available_cols = [c for c in cols if c in df.columns]
    
    return df.iloc[sim_indices][available_cols]

def recommend_hybrid(book_title, df, cosine_sim, top_n=5):
    """
    Generates book recommendations using a Hybrid approach (Content + Clustering).
    Books in the same cluster as the target book get a similarity boost.
    """
    book_title = book_title.lower()
    if book_title not in df['Book_Name_x'].values:
        return None
    
    idx = df[df['Book_Name_x'] == book_title].index[0]
    target_cluster = df.iloc[idx]['cluster']
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Boost scores for books in the same cluster (if not noise cluster -1)
    hybrid_scores = []
    for i, score in sim_scores:
        if i == idx:
            hybrid_scores.append((i, -1)) # ignore self
            continue
        boost = 0.2 if target_cluster != -1 and df.iloc[i]['cluster'] == target_cluster else 0.0
        hybrid_scores.append((i, score + boost))
        
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in hybrid_scores[:top_n]]

    cols = ['Book_Name_x', 'Author_x', 'Rating_x', 'Description', 'Ranks and Genre']
    available_cols = [c for c in cols if c in df.columns]
    
    return df.iloc[sim_indices][available_cols]

def recommend_by_genre(genre, df, top_n=5):
    """
    Cold-start recommendation: Recommends top-rated books for a given genre.
    """
    # Filter by genre (case insensitive)
    mask = df['Ranks and Genre'].str.contains(genre, case=False, na=False)
    genre_df = df[mask]
    
    if genre_df.empty:
        return None
        
    # Standardize sort columns (try to rank by Rating if available)
    sort_cols = []
    for col in ['Rating_x', 'Rating']:
        if col in genre_df.columns:
            genre_df.loc[:, col] = pd.to_numeric(genre_df[col], errors='coerce')
            sort_cols.append(col)
            break
            
    if sort_cols:
        top_books = genre_df.sort_values(by=sort_cols, ascending=False)
    else:
        top_books = genre_df
    
    cols = ['Book_Name_x', 'Author_x', 'Rating_x', 'Description', 'Ranks and Genre']
    available_cols = [c for c in cols if c in df.columns]
    
    return top_books.head(top_n)[available_cols]
