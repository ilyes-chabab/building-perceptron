import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Cleaning Report", layout="wide")

st.title("Data Cleaning & Exploratory Data Analysis Tool")
st.markdown("Upload a CSV file to generate a complete data quality and exploratory analysis report.")

# ==============================
# Upload file
# ==============================

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("File successfully loaded.")

    # ==============================
    # Data Preview
    # ==============================

    st.header("Data Preview")
    st.dataframe(df, use_container_width=True)

    # ==============================
    # Dataset Overview
    # ==============================

    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicates", df.duplicated().sum())

    # ==============================
    # Missing Values
    # ==============================

    st.header("Missing Values Analysis")

    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    missing_df = pd.DataFrame({
        "Missing Values": missing,
        "Percentage (%)": missing_percent
    }).sort_values(by="Percentage (%)", ascending=False)

    st.dataframe(missing_df)

    fig, ax = plt.subplots()
    missing_percent.sort_values(ascending=False).plot(kind='bar', ax=ax)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Missing Values per Column")
    st.pyplot(fig)

    # ==============================
    # Data Types
    # ==============================

    st.header("Data Types")
    dtype_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
    st.dataframe(dtype_df)

    # ==============================
    # Numeric Analysis
    # ==============================

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:

        st.header("Numeric Columns Analysis")

        desc = df[numeric_cols].describe().T
        st.dataframe(desc)

        # Distribution plots
        st.subheader("Distribution of Numeric Variables")

        selected_hist = st.selectbox(
            "Select a numeric column to visualize distribution",
            numeric_cols
        )

        fig_hist, ax_hist = plt.subplots()
        df[selected_hist].hist(ax=ax_hist, bins=30)
        ax_hist.set_title(f"Distribution of {selected_hist}")
        st.pyplot(fig_hist)

        # Outlier detection
        st.subheader("Outliers Detection (IQR Method)")

        outlier_percentages = {}

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_percentages[col] = (len(outliers) / len(df)) * 100

        outlier_df = pd.DataFrame.from_dict(
            outlier_percentages,
            orient="index",
            columns=["Outlier Percentage (%)"]
        ).sort_values(by="Outlier Percentage (%)", ascending=False)

        st.dataframe(outlier_df)

        # Correlation matrix
        st.subheader("Correlation Matrix")

        corr_matrix = df[numeric_cols].corr()

        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        cax = ax_corr.matshow(corr_matrix)
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        fig_corr.colorbar(cax)
        st.pyplot(fig_corr)

        # Detect high correlations
        st.subheader("High Correlations (>|0.8|)")

        high_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        if high_corr:
            high_corr_df = pd.DataFrame(
                high_corr,
                columns=["Feature 1", "Feature 2", "Correlation"]
            )
            st.dataframe(high_corr_df)
        else:
            st.write("No high correlations detected.")

    # ==============================
    # Categorical Analysis
    # ==============================

    categorical_cols = df.select_dtypes(include=["object"]).columns

    if len(categorical_cols) > 0:

        st.header("Categorical Columns Analysis")

        selected_cat = st.selectbox(
            "Select categorical column",
            categorical_cols
        )

        value_counts = df[selected_cat].value_counts(normalize=True) * 100

        st.dataframe(value_counts)

        fig_cat, ax_cat = plt.subplots()
        value_counts.head(10).plot(kind="bar", ax=ax_cat)
        ax_cat.set_ylabel("Percentage (%)")
        ax_cat.set_title(f"Top categories - {selected_cat}")
        st.pyplot(fig_cat)

        # Cardinality
        st.subheader("Cardinality Analysis")
        cardinality = df[categorical_cols].nunique().sort_values(ascending=False)
        st.dataframe(cardinality)

    # ==============================
    # Target Analysis (Optional)
    # ==============================

    st.header("Optional Target Analysis")

    target = st.selectbox("Select a target column (optional)", [None] + list(df.columns))

    if target:

        if target in numeric_cols:

            st.subheader("Correlation with Target")

            target_corr = corr_matrix[target].sort_values(ascending=False)
            st.dataframe(target_corr)

        else:

            st.subheader("Target Distribution")
            st.dataframe(df[target].value_counts())

else:
    st.info("Please upload a CSV file to begin.")
