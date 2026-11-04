"""train.py - Entrenamiento reproducible (PDF §11).

Uso:
    python train.py                                   # dataset por defecto (Churn Iran)
    python train.py --data otro.csv --target clase    # cualquier CSV de clasificación
    python train.py --drop "ID" "Otra columna" --model logistic_regression
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

import ml_core

DATA_PATH = Path("data/datos.csv")
DATA_URL = (
    "https://raw.githubusercontent.com/wilmerpineda/machine_learning_ii_course_public/"
    "main/datasets/public/Customer_Churn_Iran.csv"
)
MODEL_PATH = Path("model.joblib")
DEFAULT_TARGET = "Churn"
DEFAULT_DROP = ["Customer Value"]  # derivada del uso -> riesgo de fuga de información


def load_data(path: Path) -> pd.DataFrame:
    if path.exists():
        return ml_core.read_csv_auto(path)
    print(f"{path} no existe; descargando desde {DATA_URL}")
    return ml_core.read_csv_auto(DATA_URL)


def train_model(df: pd.DataFrame, target: str, drop: list[str], model_name: str, name: str) -> dict:
    artifact = ml_core.train(df, target=target, drop_columns=drop, model_name=model_name, dataset_name=name)
    ev = artifact["evaluation"]
    print("Variables numéricas:", artifact["numeric_features"])
    print("Variables categóricas:", artifact["categorical_features"])
    print("Matriz de confusión (filas=real, columnas=predicho):", ev["labels"])
    for row in ev["confusion_matrix"]:
        print("   ", row)
    print(ev["classification_report"])
    print(json.dumps({k: round(v, 4) for k, v in artifact["metrics"].items()}, indent=2))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el pipeline supervisado y guarda model.joblib")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--drop", nargs="*", default=None,
                        help="Columnas a excluir (por defecto: sugeridas + Customer Value si existe)")
    parser.add_argument("--model", default="random_forest", choices=list(ml_core.MODELS))
    parser.add_argument("--out", type=Path, default=MODEL_PATH)
    args = parser.parse_args()

    df = load_data(args.data)
    if args.drop is None:
        drop = ml_core.suggest_drop_columns(df, args.target)
        drop += [c for c in DEFAULT_DROP if c in df.columns and c not in drop]
    else:
        drop = args.drop
    print(f"Datos: {df.shape} | objetivo: {args.target} | excluidas: {drop} | modelo: {args.model}")

    artifact = train_model(df, args.target, drop, args.model, args.data.name)
    joblib.dump(artifact, args.out)
    assert ml_core.check_parity(joblib.load(args.out)) <= 1e-9, "Fallo de paridad tras recargar"
    print(f"Modelo guardado en {args.out}")


if __name__ == "__main__":
    main()

