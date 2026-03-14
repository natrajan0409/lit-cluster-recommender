import pandas as pd
import numpy as np
from src.recommender import load_recommender, recommend

def calculate_precision_at_k(df, cosine_sim, k=5, sample_size=50):
    """
    Calculates precision at K based on Genre overlap.
    If a recommended book share at least one genre with the query book, it's a hit.
    """
    precisions = []
    recalls = []
    
    # Sample books to evaluate
    sample_indices = np.random.choice(df.index, size=min(sample_size, len(df)), replace=False)
    
    for idx in sample_indices:
        book_title = df.iloc[idx]['Book_Name_x']
        # Extract genres, stripping whitespace and ignore rank strings (starting with #)
        genre_str = str(df.iloc[idx].get('Ranks and Genre', ''))
        query_genres = {g.strip().lower() for g in genre_str.split(',') if g.strip() and not g.strip().startswith('#')}
        
        if not query_genres:
            continue
            
        recommendations = recommend(book_title, df, cosine_sim, top_n=k)
        
        if recommendations is not None:
            hits = 0
            for _, rec in recommendations.iterrows():
                rec_genre_str = str(rec.get('Ranks and Genre', ''))
                rec_genres = {g.strip().lower() for g in rec_genre_str.split(',') if g.strip() and not g.strip().startswith('#')}
                if query_genres.intersection(rec_genres):
                    hits += 1
            
            precisions.append(hits / k)
            # Rough Recall estimation: hits relative to common genres (just for trend tracking)
            recalls.append(hits / len(query_genres) if len(query_genres) > 0 else 0)
            
    return np.mean(precisions) if precisions else 0, np.mean(recalls) if recalls else 0

if __name__ == "__main__":
    CSV_PATH = r"D:\workspace\lit-cluster-recommender\cleandata\merged_auduobook.csv"
    print("Loading system for evaluation...")
    df, cosine_sim = load_recommender(CSV_PATH)
    
    print("Calculating metrics at 5 (based on Genre overlap)...")
    avg_precision, avg_recall = calculate_precision_at_k(df, cosine_sim, k=5)
    
    print(f"\nAverage Precision@5: {avg_precision:.2%}")
    print(f"Average Recall@5 (Genre-based): {avg_recall:.2%}")
    print("\nEvaluation Complete.")
