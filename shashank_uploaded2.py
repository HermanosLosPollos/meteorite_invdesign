import pandas as pd
import numpy as np

file = "imi_meteorites_converted.xlsx"
df = pd.read_excel(file)

df.columns = df.columns.str.strip()

elements = ["Cr","Co","Ni","Cu","Ga","Ge","As","Sb","W","Re","Ir","Pt","Au"]

missing = [col for col in elements if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in elements:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=elements, how="all").reset_index(drop=True)

def compute_features(row):
    vals = row[elements].values.astype(float)
    vals = vals[~np.isnan(vals)]

    if len(vals) == 0:
        return pd.Series([np.nan]*9)

    sorted_vals = np.sort(vals)
    total = np.sum(vals)
    mean_val = np.mean(vals)

    second_max = sorted_vals[-2] if len(sorted_vals) > 1 else np.nan
    second_dom_pct = second_max / total if total != 0 else np.nan

    mad = np.mean(np.abs(vals - mean_val))

    sum_sq = np.sum(vals**2)

    rms = np.sqrt(np.mean(vals**2))

    dev_profile = mad

    variability = np.sum(np.abs(np.diff(sorted_vals)))

    count_zero = np.sum(vals < 1e-6)

    if total != 0:
        p = vals / total
        entropy = -np.sum(p * np.log(p + 1e-10))
    else:
        entropy = np.nan

    n = len(vals)
    if total != 0:
        diff_sum = np.sum(np.abs(vals[:, None] - vals))
        gini = diff_sum / (2 * n * total)
    else:
        gini = np.nan

    return pd.Series([
        second_dom_pct,
        mad,
        sum_sq,
        rms,
        dev_profile,
        variability,
        count_zero,
        entropy,
        gini
    ])

df[[
    "Second_Dominant_%", 
    "MAD",
    "Sum_of_Squares",
    "RMS",
    "Deviation_Profile",
    "Variability_Path",
    "Count_Near_Zero",
    "Entropy",
    "Gini_Coefficient"
]] = df.apply(compute_features, axis=1)

df.to_excel("Final_Selected_Features.xlsx", index=False)