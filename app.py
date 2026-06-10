import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("Customer Segmentation Dashboard")

st.write("""
This dashboard segments customers using the K-Means Clustering algorithm
based on Annual Income and Spending Score.
""")

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

if uploaded_file:

    # Read dataset
    df = pd.read_csv(uploaded_file)

    # Total Customers
    st.metric("Total Customers", len(df))

    # Dataset Preview
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Features for clustering
    X = df[['Annual Income (k$)',
            'Spending Score (1-100)']]

    # Load saved model
    model = joblib.load("models/kmeans_model.pkl")

    # Predict clusters
    df['Cluster'] = model.predict(X)

    # Clustered Customers
    st.subheader("Clustered Customers")
    st.write(df.head())

    # Customer Segments Graph
    st.subheader("Customer Segments")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=df['Cluster']
    )

    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    ax.set_title("Customer Segmentation")

    st.pyplot(fig)

    # Cluster Count
    st.subheader("Cluster Count")

    st.bar_chart(
        df['Cluster'].value_counts()
    )

    # Income and Spending Analysis
    st.subheader("Income and Spending Analysis")

    cluster_summary = df.groupby('Cluster')[[
        'Annual Income (k$)',
        'Spending Score (1-100)'
    ]].mean()

    st.write(cluster_summary)

    # Business Insights
    st.subheader("Business Insights")

    st.success(
        "Cluster 1 : High Income, High Spending (Premium Customers)"
    )

    st.info(
        "Cluster 0 : Average Income, Average Spending"
    )

    st.warning(
        "Cluster 3 : High Income, Low Spending"
    )

    st.info(
        "Cluster 2 : Low Income, High Spending"
    )

    st.error(
        "Cluster 4 : Low Income, Low Spending"
    )

    # Bonus Model Section
    st.subheader("Bonus Model: Hierarchical Clustering")

    st.write("""
    Hierarchical Clustering was implemented as an additional clustering technique.
    A dendrogram was used to analyze customer relationships and compare the
    results with K-Means clustering.
    """)

    # Model Comparison
    comparison_data = pd.DataFrame({
        "Feature": [
            "Speed",
            "Dendrogram",
            "Large Dataset Handling",
            "Customer Segmentation"
        ],
        "K-Means": [
            "Fast",
            "No",
            "Good",
            "Good"
        ],
        "Hierarchical": [
            "Slower",
            "Yes",
            "Moderate",
            "Good"
        ]
    })

    st.subheader("Model Comparison")
    st.table(comparison_data)