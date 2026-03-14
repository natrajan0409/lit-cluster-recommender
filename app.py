import streamlit as st
import pandas as pd
from src.recommender import load_recommender, recommend, recommend_hybrid, recommend_by_genre
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
st.write("Discover your next favorite listen based on NLP and Machine Learning Clustering.")

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

def display_results(results, title_message):
    if results is not None and not results.empty:
        st.success(title_message)
        for idx, row in results.iterrows():
            with st.container():
                st.markdown(f"""
                    <div class="recommendation-card">
                        <div class="book-title">{str(row['Book_Name_x']).title()}</div>
                        <div class="author-name">by {row.get('Author_x', 'Unknown Author')}</div>
                        <div style="color: #4b6cb7; font-size: 0.8rem; margin-top: 5px;">
                            ⭐ Rating: {row.get('Rating_x', 'N/A')} | Genres: {row.get('Ranks and Genre', 'N/A')}
                        </div>
                        <p style="margin-top:10px;">{str(row.get('Description', 'No description available.'))[:250]}...</p>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No books found matching the criteria. Please try another one.")

if df is not None:
    # --- Sidebar - Accuracy Metrics ---
    with st.sidebar:
        st.header("📊 Model Metrics")
        st.info("""
        **Precision @ 5**: 53.20%  
        **Recall @ 5**: 255.87%  
        *(Measured by Genre consistency and thematic overlap)*
        """)
        st.divider()
        st.write("**Methodologies Available**:")
        st.caption("- Content-Based (TF-IDF + Cosine)")
        st.caption("- Hybrid (Content + DBSCAN Clustering)")
        st.caption("- Cold-Start Genre Search")

    # --- Tabs for different features ---
    tab1, tab2, tab3, tab4 = st.tabs(["📚 Content-Based Search", "🤖 Hybrid Recommendation", "🎯 Genre Search", "📈 EDA Dashboard"])

    all_books = sorted(df['Book_Name_x'].astype(str).unique())

    # Tab 1: Content Based
    with tab1:
        st.subheader("Find Similar Books (Content-Based)")
        c_book = st.selectbox("Search for a book you liked:", all_books, key="c_book")
        c_num = st.slider("How many recommendations?", 3, 10, 5, key="c_num")
        if st.button("Generate Content Recommendations"):
            with st.spinner("Finding similar books using NLP..."):
                results = recommend(c_book, df, cosine_sim, top_n=c_num)
                display_results(results, f"Top {len(results) if results is not None else 0} Content-Based recommendations:")

    # Tab 2: Hybrid
    with tab2:
        st.subheader("Find Similar Books (Hybrid: NLP + Clustering)")
        h_book = st.selectbox("Search for a book you liked:", all_books, key="h_book")
        h_num = st.slider("How many recommendations?", 3, 10, 5, key="h_num")
        if st.button("Generate Hybrid Recommendations"):
            with st.spinner("Finding similar books using Hybrid model..."):
                results = recommend_hybrid(h_book, df, cosine_sim, top_n=h_num)
                display_results(results, f"Top {len(results) if results is not None else 0} Hybrid recommendations:")

    # Tab 3: Genre Search (Cold Start)
    with tab3:
        st.subheader("Discover by Genre (Cold Start)")
        # Extract unique genres roughly
        genres_sample = ['Fiction', 'Fantasy', 'Science Fiction', 'Thriller', 'Mystery', 'Romance', 'Nonfiction', 'Biography', 'History', 'Self Development', 'Business']
        g_genre = st.selectbox("Select a Genre:", genres_sample)
        g_num = st.slider("How many recommendations?", 3, 10, 5, key="g_num")
        if st.button("Find Top Books by Genre"):
            with st.spinner(f"Finding top {g_genre} books..."):
                results = recommend_by_genre(g_genre, df, top_n=g_num)
                display_results(results, f"Top {len(results) if results is not None else 0} books in '{g_genre}':")

    # Tab 4: EDA
    with tab4:
        st.subheader("Exploratory Data Analysis (EDA)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Authors by Book Count**")
            author_counts = df['Author_x'].value_counts().head(10)
            st.bar_chart(author_counts)
            
        with col2:
            st.markdown("**Distribution of Clusters (DBSCAN)**")
            cluster_counts = df['cluster'].value_counts()
            st.bar_chart(cluster_counts)
            
        st.markdown("**Top Books by Total Reviews**")
        # Ensure numeric for sorting
        df['Number of Reviews_x'] = pd.to_numeric(df['Number of Reviews_x'].astype(str).str.replace(',', ''), errors='coerce')
        top_reviewed = df.nlargest(10, 'Number of Reviews_x')[['Book_Name_x', 'Number of Reviews_x']].set_index('Book_Name_x')
        st.bar_chart(top_reviewed)

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Built for the Intelligent Book Recommendation Requirement</p>", unsafe_allow_html=True)
