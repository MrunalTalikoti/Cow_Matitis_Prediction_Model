
from __future__ import annotations
import pandas as pd
def load_mastitis_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
    if 'milk_visibility' in df.columns:
        df['milk_visibility'] = df['milk_visibility'].astype(str).str.strip().str.capitalize()
    return df
