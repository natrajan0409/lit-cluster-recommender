from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.cluster import DBSCAN
import pandas as pd

# Load data
df = pd.read_csv(r"D:\workspace\lit-cluster-recommender\cleandata\merged_auduobook.csv")
df['Book_Name_x'] = df['Book_Name_x'].str.lower()

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Book_Name_x'])

# Calculate Similarity (Needed for the recommend function)
cosine_sim = linear_kernel(X, X)

# DBSCAN Clustering (Your existing logic, integrated)
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
cluster_labels = dbscan.fit_predict(X)
df['cluster'] = cluster_labels

# Fixed Recommendation Function
def recommend(book_title, df, cosine_sim, top_n=5):
    book_title = book_title.lower()
    if book_title not in df['Book_Name_x'].values:
        return f"Book '{book_title}' not found in dataset."
    
    idx = df[df['Book_Name_x'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx])) # Corrected from bookname
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:top_n+1]]

    return df.iloc[sim_indices][['Book_Name_x', 'Description']]

# Run it
sample_book = df['Book_Name_x'].iloc[0]
print(f"Recommendations for: {sample_book}")
print(recommend(sample_book, df, cosine_sim))
print("\nCluster Labels:", cluster_labels)
