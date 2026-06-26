import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from recommender import load_recommender, recommend

def validate_nlp_similarity(df, cosine_sim, sample_size=50):
    """Layer 1: Direct TF-IDF Cosine Similarity Validation"""
    similarities = []
    sample_indices = np.random.choice(df.index, size=min(sample_size, len(df)), replace=False)
    
    for idx in sample_indices:
        sim_scores = cosine_sim[idx]
        top_5_indices = np.argsort(sim_scores)[-6:-1][::-1]
        top_5_sims = sim_scores[top_5_indices]
        similarities.extend(top_5_sims)
    
    return {
        'avg_similarity': np.mean(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
        'interpretation': "Higher is better (>0.6 = good NLP)"
    }

def validate_cluster_quality(X, cluster_labels):
    """Layer 2: DBSCAN Cluster Quality (Silhouette Score)"""
    # Filter out noise points (cluster == -1)
    valid_mask = cluster_labels != -1
    if sum(valid_mask) < 2:
        return {'silhouette': None, 'interpretation': 'Too many noise points'}
    
    silhouette = silhouette_score(
        X[valid_mask], 
        cluster_labels[valid_mask], 
        metric='cosine'
    )
    
    return {
        'silhouette_score': silhouette,
        'interpretation': f"Score {silhouette:.3f}: {'Good' if silhouette > 0.5 else 'Poor'} clustering"
    }

def validate_genre_overlap(df, cosine_sim, k=5, sample_size=50):
    """Layer 3: Genre-Based Validation (Existing)"""
    precisions = []
    sample_indices = np.random.choice(df.index, size=min(sample_size, len(df)), replace=False)
    
    for idx in sample_indices:
        genre_str = str(df.iloc[idx].get('Ranks and Genre', ''))
        query_genres = {g.strip().lower() for g in genre_str.split(',') 
                       if g.strip() and not g.strip().startswith('#')}
        
        if not query_genres:
            continue
        
        recommendations = recommend(df.iloc[idx]['Book_Name_x'], df, cosine_sim, top_n=k)
        
        if recommendations is not None:
            hits = sum(1 for _, rec in recommendations.iterrows() 
                      if query_genres.intersection({g.strip().lower() for g in 
                                                   str(rec.get('Ranks and Genre', '')).split(',')}))
            precisions.append(hits / k)
    
    return {
        'genre_precision@5': np.mean(precisions) if precisions else 0,
        'interpretation': "Baseline metric only - genre overlap ≠ narrative similarity"
    }

if __name__ == "__main__":
    CSV_PATH = r"D:\workspace\lit-cluster-recommender\cleandata\merged_auduobook.csv"
    print("=" * 60)
    print("COMPREHENSIVE VALIDATION REPORT")
    print("=" * 60)
    
    df, cosine_sim = load_recommender(CSV_PATH)
    
    # Load raw vectors for clustering validation
    from sklearn.feature_extraction.text import TfidfVectorizer
    df['metadata'] = (
        (df['Book_Name_x'].fillna('') + ' ') * 2 + 
        df['Description'].fillna('') + ' ' + 
        (df['Ranks and Genre'].fillna('') + ' ') * 3
    ).str.lower()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['metadata'])
    
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    cluster_labels = dbscan.fit_predict(X)
    
    print("\n📊 LAYER 1: Direct NLP Similarity (TF-IDF)")
    nlp_val = validate_nlp_similarity(df, cosine_sim)
    print(f"  Avg TF-IDF Similarity: {nlp_val['avg_similarity']:.4f}")
    print(f"  Range: [{nlp_val['min_similarity']:.4f}, {nlp_val['max_similarity']:.4f}]")
    print(f"  ✓ {nlp_val['interpretation']}")
    
    print("\n📊 LAYER 2: Cluster Quality (Silhouette)")
    cluster_val = validate_cluster_quality(X, cluster_labels)
    if cluster_val['silhouette_score'] is not None:
        print(f"  Silhouette Score: {cluster_val['silhouette_score']:.4f}")
        print(f"  ✓ {cluster_val['interpretation']}")
    else:
        print(f"  ✗ {cluster_val['interpretation']}")
    
    print("\n📊 LAYER 3: Genre-Based Validation (Baseline)")
    genre_val = validate_genre_overlap(df, cosine_sim)
    print(f"  Genre Overlap Precision@5: {genre_val['genre_precision@5']:.2%}")
    print(f"  ⚠ {genre_val['interpretation']}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("=" * 60)
    print(f"✓ NLP Quality: {nlp_val['avg_similarity']:.2f}/1.0")
    print(f"✓ Cluster Quality: {cluster_val.get('silhouette_score', 'N/A')}")
    print(f"✓ Genre Coherence: {genre_val['genre_precision@5']:.2%}")
    print("\nRecommendation: Use all three layers for confidence.")