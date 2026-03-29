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
# 3. SELECT ELEMENTS
# =========================
elements = ["Ge", "As", "Sb", "W"]

missing = [col for col in elements if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Convert to numeric
for col in elements:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows where all 4 are missing
df = df.dropna(subset=elements, how="all").reset_index(drop=True)

# =========================
# 4. NORMALIZATION
# =========================
df["Total_4"] = df[elements].sum(axis=1)
df["Total_4"] = df["Total_4"].replace(0, np.nan)

for col in elements:
    df[col + "_norm"] = df[col] / df["Total_4"]

norm_elements = [col + "_norm" for col in elements]

# =========================
# 5. FEATURE EXTRACTION
# =========================
df["Mean"] = df[norm_elements].mean(axis=1)
df["Median"] = df[norm_elements].median(axis=1)
df["Standard Deviation"] = df[norm_elements].std(axis=1)
df["Range"] = df[norm_elements].max(axis=1) - df[norm_elements].min(axis=1)
df["Coefficient of Variation (CV)"] = df["Standard Deviation"] / df["Mean"]

def entropy(row):
    vals = pd.to_numeric(row[norm_elements], errors="coerce").dropna().astype(float).values
    if len(vals) == 0:
        return np.nan
    total = vals.sum()
    if total == 0:
        return np.nan
    p = vals / total
    p = p[p > 0]
    return -(p * np.log(p)).sum()

df["Entropy"] = df.apply(entropy, axis=1)

def safe_skew(row):
    vals = pd.to_numeric(row[norm_elements], errors="coerce").dropna().astype(float).values
    if len(vals) < 3:
        return np.nan
    return skew(vals)

df["Skewness"] = df.apply(safe_skew, axis=1)

def safe_kurtosis(row):
    vals = pd.to_numeric(row[norm_elements], errors="coerce").dropna().astype(float).values
    if len(vals) < 4:
        return np.nan
    return kurtosis(vals)

df["Kurtosis"] = df.apply(safe_kurtosis, axis=1)

df["Dominant Element"] = df[norm_elements].astype(float).idxmax(axis=1).str.replace("_norm", "", regex=False)
df["Top Element Percentage"] = df[norm_elements].astype(float).max(axis=1)

# =========================
# 6. SAVE OUTPUT AS CSV
# =========================
output_file = "final_10_features.csv"

final_cols = elements + norm_elements + [
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

df[final_cols].to_csv(output_file, index=False)

print("✅ Done! Saved as:", output_file)
print(df[final_cols].head())