
from __future__ import annotations
import pandas as pd, numpy as np
from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def split_features_target(df: pd.DataFrame, target_col: str, drop_cols: List[str]|None=None):
    drop_cols = drop_cols or []
    keep_cols = [c for c in df.columns if c not in set(drop_cols+[target_col])]
    X = df[keep_cols].copy(); y = df[target_col].astype(int).copy()
    return X, y

def build_preprocessor(X: pd.DataFrame):
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]), num),
        ("cat", Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]), cat)
    ])
    return pre, num, cat
