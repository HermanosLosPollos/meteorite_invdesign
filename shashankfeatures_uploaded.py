import pandas as pd
import numpy as np

# =========================
# 1. LOAD FILE
# =========================
file = "imi_meteorites_maindataset.xlsx"
df = pd.read_excel(file)

# =========================
# 2. CLEAN COLUMNS
# =========================
df.columns = df.columns.str.strip()

# =========================
# 3. DEFINE ALL 13 ELEMENTS
# =========================
elements = ["Cr","Co","Ni","Cu","Ga","Ge","As","Sb","W","Re","Ir","Pt","Au"]

# check columns exist
missing = [col for col in elements if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# convert to numeric
for col in elements:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# drop rows where all are missing
df = df.dropna(subset=elements, how="all").reset_index(drop=True)

# =========================
# 4. SORT VALUES
# =========================
def sort_vals(row):
    vals = row[elements].values.astype(float)
    vals = np.sort(vals)
    return pd.Series(vals)

sorted_cols = [f"v{i}" for i in range(13)]
df[sorted_cols] = df.apply(sort_vals, axis=1)

# =========================
# 5. FEATURES (ALL 13)
# =========================

# Max & Min
df["Max_13"] = df["v12"]
df["Min_13"] = df["v0"]

# Second highest & second lowest
df["Second_Highest_13"] = df["v11"]
df["Second_Lowest_13"] = df["v1"]

# Top 3 & Bottom 3
df["Top3_Sum"] = df[["v10","v11","v12"]].sum(axis=1)
df["Bottom3_Sum"] = df[["v0","v1","v2"]].sum(axis=1)

df["Top3_Avg"] = df["Top3_Sum"] / 3
df["Bottom3_Avg"] = df["Bottom3_Sum"] / 3

# Differences
df["Max_Min_Diff"] = df["v12"] - df["v0"]
df["Top3_Bottom3_Diff"] = df["Top3_Sum"] - df["Bottom3_Sum"]

# =========================
# 6. SAVE OUTPUT
# =========================
df.to_excel("All13_Cumulative_Features.xlsx", index=False)

print("✅ Done! Cumulative features for all 13 elements extracted.")
print("Saved as: All13_Cumulative_Features.xlsx")