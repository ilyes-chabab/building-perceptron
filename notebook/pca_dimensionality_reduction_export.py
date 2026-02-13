
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==============================
# Configuration
# ==============================

INPUT_FILE = "bcw_data.csv"   # Replace with your input CSV file path
OUTPUT_FILE = "bcw_data_pca_reduced.csv"
VARIANCE_THRESHOLD = 0.95  # Keep 95% of explained variance

# ==============================
# Load dataset
# ==============================

df = pd.read_csv(INPUT_FILE)

# ==============================
# Prepare data
# ==============================

# Encode target variable
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

# Separate features and target
X = df.drop(columns=["id", "diagnosis"])
y = df["diagnosis"]
ids = df["id"]

# ==============================
# Standardization (required for PCA)
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# PCA
# ==============================

pca = PCA(n_components=VARIANCE_THRESHOLD)
X_pca = pca.fit_transform(X_scaled)

print("Original number of features:", X.shape[1])
print("Reduced number of components:", X_pca.shape[1])
print("Explained variance ratio sum:", pca.explained_variance_ratio_.sum())

# ==============================
# Create reduced dataframe
# ==============================

pca_columns = [f"PC{i+1}" for i in range(X_pca.shape[1])]

df_reduced = pd.DataFrame(X_pca, columns=pca_columns)

# Add back id and target
df_reduced.insert(0, "diagnosis", y.values)
df_reduced.insert(0, "id", ids.values)

# ==============================
# Export CSV
# ==============================

df_reduced.to_csv(OUTPUT_FILE, index=False)

print(f"Reduced dataset successfully saved to {OUTPUT_FILE}")
