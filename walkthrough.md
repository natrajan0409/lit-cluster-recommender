# Technical Walkthrough: Audiobook Data Processing

This walkthrough provides a detailed look into the current implementation of the `lit-cluster-recommender` project, focusing on the data cleaning and matching pipeline.

## 🏗️ Modular Architecture

The project is designed with modularity in mind, separating concerns into distinct Python scripts within the `src/` directory:

- **[data_loader.py](file:///d:/workspace/lit-cluster-recommender/src/data_loader.py)**: Handles loading disparate CSV datasets (Base Catalog and Advanced Features) into Pandas DataFrames.
- **[data_cleaner.py](file:///d:/workspace/lit-cluster-recommender/src/data_cleaner.py)**: Provides a standard interface for deduplication and basic cleaning tasks.

## 🧵 Data Processing Pipeline

The core logic resides in [dataclean.ipynb](file:///d:/workspace/lit-cluster-recommender/dataclean.ipynb), which follows these steps:

### 1. Data Ingestion
Datasets are loaded from the `DATASET/` directory using the [DataLoader](file:///d:/workspace/lit-cluster-recommender/src/data_loader.py#4-15) class. Initial row counts are verified to ensure data integrity.

### 2. Schema Standardization
Columns are renamed (e.g., `Book Name` to `Book_Name`) to provide a consistent key for merging.

### 3. Optimized Fuzzy Matching
One of the key technical achievements in this phase is the resolution of book title inconsistencies.
- **Problem**: Book titles across datasets often have slight differences in subtitles, punctuation, or casing. 
- **Solution**: We use `rapidfuzz` for high-performance string matching.
- **Optimization**: To save computation time, we only perform fuzzy matching on titles that do not have an exact string match. This significantly reduces the search space.
- **Thresholding**: A `fuzz.WRatio` score of `>= 90` is used to ensure high-confidence matches.

```python
# snippet of the fuzzy matching logic
matches = {}
for book_name in tqdm(unmatched_books):
    match = process.extractOne(book_name, base_catalog_names, scorer=fuzz.WRatio)
    if match and match[1] >= 90:
        matches[book_name] = match[0]
```

### 4. Unified Data Merging
The fuzzy match results are mapped back to the dataframes, allowing for an inner join that combines the rich metadata of the "Advanced Features" catalog with the "Base" catalog.

## Output
The final cleaned and merged dataset is saved to [cleandata/merged_auduobook.csv](file:///d:/workspace/lit-cluster-recommender/cleandata/merged_auduobook.csv), containing both base info and advanced features for over 4,300 unique titles.

## 🔭 Next Steps
- **Text Processing**: Tokenizing and cleaning book descriptions for NLP.
- **Clustering**: Applying K-Means or similar models to group books by features and genres.
- **Recommendation UI**: Building the Streamlit interface to allow users to interact with the model.
