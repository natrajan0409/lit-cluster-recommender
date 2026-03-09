import pandas as pd
import numpy as np
from src.recommender import load_recommender, recommend

def calculate_precision_at_k(df, cosine_sim, k=5, sample_size=50):
    """
    Calculates precision at K based on Genre overlap.
    If a recommended book share at least one genre with the query book, it's a hit.
    """
    precisions = []
    
    # Sample books to evaluate
    sample_indices = np.random.choice(df.index, size=min(sample_size, len(df)), replace=False)
    
    for idx in sample_indices:
        book_title = df.iloc[idx]['Book_Name_x']
        query_genres = set(str(df.iloc[idx].get('Ranks and Genre', '')).lower().split(','))
        
        recommendations = recommend(book_title, df, cosine_sim, top_n=k)
        
        if recommendations is not None:
            hits = 0
            for _, rec in recommendations.iterrows():
                rec_genres = set(str(rec.get('Ranks and Genre', '')).lower().split(','))
                # Check for overlap
                if query_genres.intersection(rec_genres):
                    hits += 1
            
            precisions.append(hits / k)
            
    return np.mean(precisions) if precisions else 0

if __name__ == "__main__":
    CSV_PATH = r"D:\workspace\lit-cluster-recommender\cleandata\merged_auduobook.csv"
    print("Loading system for evaluation...")
    df, cosine_sim = load_recommender(CSV_PATH)
    
    print("Calculating Precision at 5 (based on Genre overlap)...")
    avg_precision = calculate_precision_at_k(df, cosine_sim, k=5)
    
    print(f"\nAverage Precision@5: {avg_precision:.2%}")
    print("\nEvaluation Complete.")
