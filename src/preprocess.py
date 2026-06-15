"""
src/preprocess.py
Funções de pré-processamento para o pipeline de classificação de vinho.
Dataset: WineQT.csv (vinho tinto, separador vírgula, coluna Id a descartar)
"""

import os
import pandas as pd
import numpy as np

def load_wine_data(data_dir: str = "../data",
                   filename: str = "WineQT.csv") -> pd.DataFrame:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    df = pd.read_csv(path, sep=",")

    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
        print("[load] Coluna 'Id' removida.")

    print(f"[load] {path}: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def create_binary_target(df: pd.DataFrame, threshold: int = 7) -> pd.DataFrame:
    df = df.copy()
    df["high_quality"] = (df["quality"] >= threshold).astype(int)
    n_high = df["high_quality"].sum()
    pct    = n_high / len(df) * 100
    print(f"[target] Alta qualidade (nota >= {threshold}): {n_high} amostras ({pct:.1f}%)")
    print(f"[target] Baixa/Média  (nota <  {threshold}): {len(df)-n_high} amostras ({100-pct:.1f}%)")
    return df

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("[missing] Nenhum valor faltante encontrado.")
        return df
    for col in missing.index:
        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[missing] '{col}': imputado com mediana={median_val:.4f}")
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["acidity_ratio"]   = df["fixed acidity"] / (df["volatile acidity"] + 1e-9)
    df["sulfur_ratio"]    = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-9)
    df["alcohol_density"] = df["alcohol"] / df["density"]
    print("[features] Criadas: acidity_ratio, sulfur_ratio, alcohol_density")
    return df


FEATURE_COLS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
    "acidity_ratio", "sulfur_ratio", "alcohol_density",
]

def full_preprocess(data_dir: str = "../data",
                    filename: str = "WineQT.csv") -> tuple:
    df = load_wine_data(data_dir, filename)
    df = handle_missing(df)
    df = create_binary_target(df)
    df = engineer_features(df)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available]
    y = df["high_quality"]

    print(f"\n[preprocess] X: {X.shape} | y: {y.shape}")
    return X, y

if __name__ == "__main__":
    X, y = full_preprocess()
    print(X.head())