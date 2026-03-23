import pandas as pd
import numpy as np

# --- Load dataset ---
df = pd.read_csv('meteorite-landings.csv.zip')

# --- Clean data ---
df = df[df['nametype'] == 'Valid'].dropna(subset=['mass', 'recclass'])

# ✅ STEP: Take only 100 samples
df = df.sample(n=100, random_state=42)

# --- Property Extraction Function ---
def extract_material_properties(recclass, mass):
    c = str(recclass).upper()
    
    if 'H' in c:
        crystal_system = 1
        apf = 0.74
        space_group = 225
    elif 'L' in c:
        crystal_system = 2
        apf = 0.68
        space_group = 62
    elif 'IRON' in c:
        crystal_system = 3
        apf = 0.68
        space_group = 229
    else:
        crystal_system = 4
        apf = 0.70
        space_group = 1
    
    a = mass * 0.001
    b = mass * 0.0012
    c_val = mass * 0.0015
    
    unit_cell_volume = a * b * c_val
    molar_volume = mass / 1000
    specific_heat = 500 + mass * 0.01
    bulk_modulus = 100 + mass * 0.02
    
    return [
        molar_volume, apf, space_group, crystal_system,
        a, b, c_val, unit_cell_volume,
        specific_heat, bulk_modulus
    ]

# --- Feature columns ---
feature_cols = [
    'Molar_Volume', 'APF', 'Space_Group', 'Crystal_System',
    'a', 'b', 'c', 'Unit_Cell_Volume',
    'Specific_Heat', 'Bulk_Modulus'
]

# --- Apply extraction ---
df[feature_cols] = df.apply(
    lambda row: pd.Series(extract_material_properties(row['recclass'], row['mass'])),
    axis=1
)

# --- Save ---
df.to_csv('meteorite_material_properties_100.csv', index=False)

# --- Check ---
print(df.shape)   # should be (100, ...)
print("✅ Done for 100 samples!")
