import pandas as pd
import numpy as np

# =========================
# 1. LOAD NEW FILE
# =========================
file = "imi_meteorites_maindataset.xlsx"

df = pd.read_excel(file)

# =========================
# 2. CLEAN COLUMN NAMES
# =========================
df.columns = df.columns.str.strip()

# =========================
# 3. SELECT ELEMENTS
# =========================
elements = ["Re", "Ir", "Pt", "Au"]

# check if columns exist
missing = [col for col in elements if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# convert to numeric
for col in elements:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# remove empty rows
df = df.dropna(subset=elements, how="all").reset_index(drop=True)

# =========================
# 4. SORT VALUES
# =========================
def sort_vals(row):
    vals = row[elements].values.astype(float)
    vals = np.sort(vals)
    return pd.Series(vals)

df[["min_val", "second_min", "second_max", "max_val"]] = df.apply(sort_vals, axis=1)

# =========================
# 5. YOUR 10 FEATURES
# =========================

df["Maximum_Value"] = df["max_val"]
df["Minimum_Value"] = df["min_val"]
df["Second_Highest_Value"] = df["second_max"]
df["Second_Lowest_Value"] = df["second_min"]

df["Top_2_Sum"] = df["max_val"] + df["second_max"]
df["Bottom_2_Sum"] = df["second_min"] + df["min_val"]

df["Top_2_Average"] = df["Top_2_Sum"] / 2
df["Bottom_2_Average"] = df["Bottom_2_Sum"] / 2

df["Top_Bottom_Difference"] = df["max_val"] - df["min_val"]
df["Top2_Bottom2_Difference"] = df["Top_2_Sum"] - df["Bottom_2_Sum"]

# =========================
# 6. SAVE OUTPUT
# =========================
df.to_excel("Re_Ir_Pt_Au_10features_final.xlsx", index=False)

print("✅ Done! Features extracted for new dataset.")
print("Saved as: Re_Ir_Pt_Au_10features_final.xlsx")