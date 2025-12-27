# scripts/eda.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_PATH = "data/raw/heart.csv"
FIGURES_PATH = "reports/figures"

os.makedirs(FIGURES_PATH, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# -----------------------------
# 1. Histograms of features
# -----------------------------
df.hist(figsize=(15, 12), bins=20)
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/histograms.png")
plt.close()

# -----------------------------
# 2. Correlation Heatmap
# -----------------------------
plt.figure(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/correlation_heatmap.png")
plt.close()

# -----------------------------
# 3. Class Balance Plot
# -----------------------------
plt.figure(figsize=(6, 4))
df["target"].value_counts().plot(kind="bar")
plt.title("Class Balance (Heart Disease)")
plt.xlabel("Target")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/class_balance.png")
plt.close()

print("EDA completed. Figures saved to reports/figures/")
