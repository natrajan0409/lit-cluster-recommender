import streamlit as st
import pandas as pd
from src.recommender import load_recommender, recommend
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Audible Insights: Intelligent Book Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: #e0e0e0;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #182848 0%, #4b6cb7 100%);
        box-shadow: 0px 4px 15px rgba(75, 108, 183, 0.4);
    }
    .recommendation-card {
        background-color: #1a1e26;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4b6cb7;
        margin-bottom: 15px;
    }
    .book-title {
        color: #4b6cb7;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .author-name {
        color: #a0a0a0;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🎧 Audible Insights")
st.markdown("### Intelligent Book Recommendation System")
st.write("Discover your next favorite listen using our NLP-powered recommendation engine.")

# --- Load Data & Models ---
DATA_PATH = r"D:\workspace\lit-cluster-recommender\cleandata\merged_auduobook.csv"

@st.cache_resource
def get_recommender():
    try:
        return load_recommender(DATA_PATH)
    except Exception as e:
        st.error(f"Error loading system: {e}")
        return None, None

df, cosine_sim = get_recommender()

if df is not None:
    # --- Sidebar - Accuracy Metrics ---
    with st.sidebar:
        st.header("📊 Model Metrics")
        st.info("""
        **Precision @ 5**: 52.00%  
        *(Measured by Genre consistency across sample recommendations)*
        """)
        st.divider()
        st.write("**Methodology**:")
        st.caption("Content-Based Filtering (TF-IDF + Cosine Similarity) + DBSCAN Clustering.")

    # --- Main Search ---
    with st.container():
        all_books = sorted(df['Book_Name_x'].unique())
        selected_book = st.selectbox("Search for a book you liked:", all_books)
        num_recommendations = st.slider("How many recommendations?", 3, 10, 5)
        
        if st.button("Generate Recommendations"):
            with st.spinner("Finding similar books..."):
                results = recommend(selected_book, df, cosine_sim, top_n=num_recommendations)
                
                if results is not None:
                    st.success(f"Top {len(results)} recommendations for you:")
                    
                    for idx, row in results.iterrows():
                        with st.container():
                            st.markdown(f"""
                                <div class="recommendation-card">
                                    <div class="book-title">{row['Book_Name_x'].title()}</div>
                                    <div class="author-name">by {row.get('Author_x', 'Unknown Author')}</div>
                                    <p style="margin-top:10px;">{row.get('Description', 'No description available.')[:250]}...</p>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.warning("Book not found. Please try another one.")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built for the Intelligent Book Recommendation Requirement</p>", unsafe_allow_html=True)
