# Technical Walkthrough: Audible Insights Recommender

This walkthrough provides a detailed look into the data processing, NLP modeling, clustering, and Streamlit presentation logic for the **Audible Insights: Intelligent Book Recommendations** project.

## 🧵 Data Processing & Merging Pipeline

The core logic for initial processing resides in `dataclean.ipynb`:

### 1. Data Ingestion & Schema Standardization
Datasets (Base Catalog and Advanced Features) are ingested using pandas. Columns are standardly renamed (e.g., `Book Name` to `Book_Name`) to provide a consistent merge key.

### 2. Optimized Fuzzy Matching (`rapidfuzz`)
To resolve book title string inconsistencies across datasets (subtitles, punctuation, missing words):
- We utilize `rapidfuzz` for high-performance string matching using `fuzz.WRatio`.
- **Optimization**: To save time, fuzzy matching is only applied to titles that lack an exact string match.
- The results are mapped back, yielding the final rich dataset: `cleandata/merged_auduobook.csv` (over 4,300 titles).

---

## 🧠 Recommendation Engine Logic (`src/recommender.py`)

The recommendation system consists of multiple models combined into a hybrid approach to improve result accuracy.

### 1. Feature Engineering & Vectorization (NLP)
We use a **Content-Based Filtering** approach by analyzing text metadata.
- **Combined Meta String**: We concatenate the `Book_Name`, `Description`, and `Ranks and Genre` columns into a single `metadata` string.
- **Weighting**: Book Titles and Genres are duplicated in the string to give them higher mathematical weight during vectorization.
- **TF-IDF Vectorization**: We use `TfidfVectorizer(stop_words='english')` to convert the text into numerical vectors, filtering out common useless english stop words.

### 2. Model 1: Content-Based Filtering (Cosine Similarity)
- We calculate the **Cosine Similarity** (`linear_kernel` for speed) across all TF-IDF vectors.
- This creates an $N \times N$ matrix where a score of 1.0 means exact text overlap, and 0.0 means no overlap.
- The `recommend()` function simply looks up the user's book in this matrix, sorts the array descending, and returns the top 5 highest-scoring adjacent books.

### 3. Clustering (DBSCAN)
- We apply `DBSCAN(eps=0.5, min_samples=5, metric='cosine')` against the TF-IDF matrix.
- DBSCAN groups books with similar themes/genres into identical integer `cluster` IDs. Outliers are flagged as `-1` (noise).

### 4. Model 2: The Hybrid Recommender
The `recommend_hybrid()` function merges the NLP and Clustering outputs.
- It iterates over the standard Cosine Similarity scores.
- **The Boost**: If a recommended book shares the exact same `cluster` ID as the query book (and the cluster is not `-1`), it adds a `+0.2` mathematical boost to the cosine score.
- This hybrid fusion pushes books that are semantically *and* structurally grouped higher up the recommendation chain.

### 5. Model 3: Cold-Start (Genre Search)
The `recommend_by_genre()` function handles new users who don't have a starting book.
- It performs a case-insensitive Pandas filter against the `Ranks and Genre` column.
- It then sorts the resulting subset by `Rating` descending, returning the top-rated books within that category. 

---

## 🖥️ Streamlit Application (`app.py`)

The full frontend is deployed using Streamlit (`streamlit run app.py`), split into a multi-tab workspace.

1. **State Persistence & Caching**
   - The dataset and matrix calculations are heavy. We decorate `get_recommender()` with `@st.cache_resource` so the TF-IDF matrix is computed exactly **once** upon server boot, drastically improving user response times.
2. **Interactive Tabs**
   - **Tabs 1 & 2**: Hooks into the content-based and hybrid `recommender.py` logic. We map the returned DataFrames into custom HTML/CSS cards (`st.markdown`) for a highly polished UI.
   - **Tab 3**: Harnesses the Cold-Start model via a generic "Genre" select-box dropdown.
   - **Tab 4 (EDA Dashboard)**: Fulfills the Exploratory Data Analysis requirement. Uses `st.bar_chart` on aggregated dataframes:
     - The top 10 Authors by Book Count `value_counts()`
     - The DBSCAN cluster size distribution
     - Most Reviewed Books (using pandas `to_numeric()` conversions to sort string numbers correctly). 
     
### 📊 Evaluation Metrics
As highlighted in the sidebar, we calculate custom metrics in `src/evaluation.py`. Because the dataset lacks User IDs (Collaborative Filtering isn't viable), we evaluate using **Precision/Recall @ 5** by verifying if the recommended books overlap in *Genre* mappings with the query book.
