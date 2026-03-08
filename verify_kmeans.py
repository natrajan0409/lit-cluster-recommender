from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

# Mock data
data = {
    'Book_Name_x': ['The Great Gatsby', '1984', 'To Kill a Mockingbird', 'The Catcher in the Rye', 'Pride and Prejudice'],
    'Author_x': ['F. Scott Fitzgerald', 'George Orwell', 'Harper Lee', 'J.D. Salinger', 'Jane Austen']
}
df = pd.DataFrame(data)

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(df['Book_Name_x'])

# KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(x)
df['cluster'] = cluster_labels

print("Cluster Labels:", cluster_labels)
print("DataFrame with Clusters:\n", df)
