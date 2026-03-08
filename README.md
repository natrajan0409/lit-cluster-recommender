# Lit-Cluster-Recommender

A modular book recommendation system that ingests datasets, cleans and processes text, and applies NLP with clustering methods to build multiple recommendation models. Users can explore intelligent book suggestions through a streamlined Streamlit interface, combining data-driven insights with an engaging user experience.

## 🚀 Features & Current Progress

- **Modular Data Ingestion**: Cleanly separated data loading logic in `src/data_loader.py`.
- **Advanced Title Matching**: Implemented high-performance fuzzy matching for book titles using the `rapidfuzz` library to resolve inconsistencies between disparate catalogs.
- **Data Merging & Deduplication**: Automated merging of advanced features with base catalog data for a unified analysis dataset.
- **Exploratory Notebooks**: Documented data cleaning process in `dataclean.ipynb`.

## 🛠️ Technical Stack

- **Data Processing**: Python, Pandas
- **Fuzzy Matching**: RapidFuzz
- **Visualization**: Matplotlib, Seaborn
- **Utilities**: TQDM (progress tracking)
- **Environment Management**: Python venv

## ⚙️ Setup & Usage

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd lit-cluster-recommender
   ```

2. **Set up the environment**:
   ```bash
   python -m venv env
   .\env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run Data Processing**:
   Open `dataclean.ipynb` in your preferred Jupyter environment and run the cells to process the raw audiobook datasets.

## 📂 Project Structure

- `src/`: Core modular Python scripts (DataLoader, DataCleaner).
- `DATASET/`: Raw CSV data sources (not tracked in Git).
- `cleandata/`: Processed output files.
- `Requirementdoc/`: Project documentation and PDFs.
- `dataclean.ipynb`: Interactive data cleaning workflow.
