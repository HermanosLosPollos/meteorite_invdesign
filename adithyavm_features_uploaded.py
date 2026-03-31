import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# =========================
# 1. LOAD FILE
# =========================
file = "imi_meteorites_maindataset.xlsx"
df = pd.read_excel(file, engine="openpyxl")

# =========================
# 2. CLEAN DATA
# =========================
df = df.dropna(how="all").dropna(axis=1, how="all")
df.columns = [str(col).strip() for col in df.columns]

# =========================
# 3. DEFINE COLUMNS
# =========================

# ---- A) ALL ELEMENTS for first 5 features ----
# If first few columns are metadata, exclude them manually
# Example:
# all_elements = df.columns[3:].tolist()

# Better automatic approach: convert all to numeric first
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

all_elements = df.select_dtypes(include=[np.number]).columns.tolist()

# Optional: remove non-element numeric metadata columns if needed
# all_elements = [col for col in all_elements if col not in ["ID", "Sample_No"]]

if len(all_elements) == 0:
    raise ValueError("No numeric element columns found for all-elements statistics!")

# ---- B) ONLY THESE 4 for advanced features ----
selected_elements = ["Ge", "As", "Sb", "W"]

missing = [col for col in selected_elements if col not in df.columns]
if missing:
    raise ValueError(f"Missing selected columns: {missing}")

# Convert selected columns again safely
for col in selected_elements:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows where all all_elements are missing
df = df.dropna(subset=all_elements, how="all").reset_index(drop=True)

# =========================
# 4. FIRST 5 FEATURES --> FOR ALL ELEMENTS
# =========================
df["Mean"] = df[all_elements].mean(axis=1)
df["Median"] = df[all_elements].median(axis=1)
df["Standard Deviation"] = df[all_elements].std(axis=1)
df["Range"] = df[all_elements].max(axis=1) - df[all_elements].min(axis=1)
df["Coefficient of Variation (CV)"] = df["Standard Deviation"] / df["Mean"]

# =========================
# 5. ADVANCED FEATURES --> ONLY FOR Ge, As, Sb, W
# =========================
df["Total_4"] = df[selected_elements].sum(axis=1)
df["Total_4"] = df["Total_4"].replace(0, np.nan)

for col in selected_elements:
    df[col + "_norm"] = df[col] / df["Total_4"]

norm_selected = [col + "_norm" for col in selected_elements]

# Entropy
def entropy(row):
    vals = pd.to_numeric(row[norm_selected], errors="coerce").dropna().astype(float).values
    if len(vals) == 0:
        return np.nan
    total = vals.sum()
    if total == 0:
        return np.nan
    p = vals / total
    p = p[p > 0]
    return -(p * np.log(p)).sum()

df["Entropy"] = df.apply(entropy, axis=1)

# Skewness
def safe_skew(row):
    vals = pd.to_numeric(row[selected_elements], errors="coerce").dropna().astype(float).values
    if len(vals) < 3:
        return np.nan
    return skew(vals)

df["Skewness"] = df.apply(safe_skew, axis=1)

# Kurtosis
def safe_kurtosis(row):
    vals = pd.to_numeric(row[selected_elements], errors="coerce").dropna().astype(float).values
    if len(vals) < 4:
        return np.nan
    return kurtosis(vals)

df["Kurtosis"] = df.apply(safe_kurtosis, axis=1)

# Dominant Element
df["Dominant Element"] = df[norm_selected].astype(float).idxmax(axis=1).str.replace("_norm", "", regex=False)

# Top Element Percentage
df["Top Element Percentage"] = df[norm_selected].astype(float).max(axis=1)

# =========================
# 6. SAVE OUTPUT
# =========================
output_file = "final_meteorite_features.csv"

final_cols = all_elements + selected_elements + norm_selected + [
    "Mean",
    "Median",
    "Standard Deviation",
    "Range",
    "Coefficient of Variation (CV)",
    "Entropy",
    "Skewness",
    "Kurtosis",
    "Dominant Element",
    "Top Element Percentage"
]

# Remove duplicate column names if selected_elements already in all_elements
final_cols = list(dict.fromkeys(final_cols))

df[final_cols].to_csv(output_file, index=False)

print("✅ Done! Saved as:", output_file)
print(df[final_cols].head())