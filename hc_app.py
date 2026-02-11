import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

st.set_page_config(page_title="News Topic Discovery Dashboard", layout="wide")

st.title("🟣 News Topic Discovery Dashboard")
st.write(
    "This system uses **Hierarchical Clustering** to automatically group similar news articles "
    "based on textual similarity."
)

st.info("👉 Discover hidden themes without defining categories upfront.")

st.sidebar.header("📂 Dataset Handling")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin1")

    possible_cols = ["text", "news", "headline", "article", "content"]
    text_col = None
    for col in df.columns:
        if col.lower() in possible_cols:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[-1]

    st.sidebar.success(f"Detected Text Column: **{text_col}**")

    st.sidebar.header("📝 Text Vectorization Controls")

    max_features = st.sidebar.slider("Maximum TF-IDF Features", 100, 2000, 1000)

    stopwords = st.sidebar.checkbox("Use English Stopwords", value=True)

    ngram_option = st.sidebar.selectbox(
        "N-gram Range",
        ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
    )

    if ngram_option == "Unigrams":
        ngram_range = (1, 1)
    elif ngram_option == "Bigrams":
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 2)

    st.sidebar.header("🌳 Hierarchical Clustering Controls")

    linkage_method = st.sidebar.selectbox(
        "Linkage Method",
        ["ward", "complete", "average", "single"]
    )

    dendro_size = st.sidebar.slider(
        "Number of Articles for Dendrogram", 20, 200, 100
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english" if stopwords else None,
        ngram_range=ngram_range
    )

    X = vectorizer.fit_transform(df[text_col].astype(str))

    st.sidebar.header("3️⃣ Clustering Control")

    if st.sidebar.button("🟦 Generate Dendrogram"):

        st.subheader("🌳 Dendrogram Visualization")

        X_subset = X[:dendro_size].toarray()
        Z = linkage(X_subset, method=linkage_method)

        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(Z, truncate_mode="level", p=5)

        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Article Index")
        ax.set_ylabel("Distance")

        st.pyplot(fig)

        st.warning(
            "Look for **large vertical gaps** — they indicate strong topic separation."
        )

    st.sidebar.header("🟩 Apply Clustering")

    num_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

    if st.sidebar.button("🟩 Apply Clustering"):

        st.subheader("📌 Clustering Results")

        model = AgglomerativeClustering(
            n_clusters=num_clusters,
            linkage=linkage_method,
            metric="euclidean"
        )

        labels = model.fit_predict(X.toarray())
        df["Cluster"] = labels

        st.subheader("📍 Cluster Visualization (PCA Projection)")

        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X.toarray())

        plot_df = pd.DataFrame({
            "PCA1": X_2d[:, 0],
            "PCA2": X_2d[:, 1],
            "Cluster": labels,
            "Snippet": df[text_col].astype(str).str[:120]
        })

        st.scatter_chart(plot_df, x="PCA1", y="PCA2", color="Cluster")

        st.subheader("📊 Cluster Summary (Business View)")

        terms = vectorizer.get_feature_names_out()
        summary = []

        for c in range(num_clusters):
            mask = (df["Cluster"].values == c)

            cluster_texts = df.loc[mask, text_col]
            size = cluster_texts.shape[0]

            if size == 0:
                continue

            cluster_matrix = X[mask].mean(axis=0)

            top_idx = np.asarray(cluster_matrix).flatten().argsort()[-10:][::-1]
            keywords = [terms[i] for i in top_idx]

            snippet = cluster_texts.iloc[0][:150]

            summary.append([c, size, ", ".join(keywords), snippet])

        summary_df = pd.DataFrame(
            summary,
            columns=["Cluster ID", "No. of Articles", "Top Keywords", "Representative Snippet"]
        )

        st.dataframe(summary_df)

        st.subheader("📊 Validation: Silhouette Score")

        sil_score = silhouette_score(X.toarray(), labels)

        st.metric("Silhouette Score", round(sil_score, 3))

        st.write("""
        **Interpretation:**
        - Close to **1** → clusters are well separated  
        - Close to **0** → clusters overlap  
        - Negative → poor clustering  
        """)

        st.subheader("📰 Editorial Insights (Human Language)")

        st.write("""
        These clusters represent natural news themes:

        - 🟣 Cluster 0: Articles likely related to financial markets and business performance  
        - 🟡 Cluster 1: Corporate earnings, company strategy, quarterly results  
        - 🔵 Cluster 2: Economic policy, government announcements, regulations  

        Editors can use these groupings for:
        - Automatic tagging  
        - Personalized recommendations  
        - Better content organization  
        """)

        st.info(
            "Articles grouped in the same cluster share similar vocabulary and themes. "
            "These clusters can be used for automatic tagging, recommendations, and content organization."
        )