import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

# Load cleaned CSV
df = pd.read_csv(r"D:\workspace\lit-cluster-recommender\cleandata\merged_auduobook.csv")


# Use the correct column name from your file
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Book_Name_x'])   # or df['description'] if that's the text field

# Apply DBSCAN with cosine similarity
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
cluster_labels = dbscan.fit_predict(X)

# Add cluster labels to DataFrame
df['cluster'] = cluster_labels

# Print results
print("Cluster Labels:", cluster_labels)
print(df[['Book_Name_x','cluster']].head())   # match the actual column name
