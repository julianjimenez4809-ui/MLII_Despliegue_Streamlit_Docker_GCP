"""ml_core.py - Lógica común de entrenamiento e inferencia (Ruta A - supervisada).

La usan train.py (entrenamiento reproducible) y app.py (Streamlit), de modo que
las transformaciones de entrenamiento y de serving sean SIEMPRE las mismas
(Pipeline de scikit-learn: preprocesamiento + modelo en un solo objeto).
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
MAX_CAT_UNIQUE = 10        # enteros con <= 10 valores distintos se tratan como categóricos
MAX_CLASSES = 20           # más clases que esto -> probablemente es regresión
MODELS = {
    "logistic_regression": "Regresión logística",
    "random_forest": "Random Forest",
}


# ----------------------------------------------------------------------------
# Lectura y limpieza
# ----------------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres: quita BOM y espacios repetidos ('Call  Failure' -> 'Call Failure')."""
    df = df.copy()
    df.columns = [" ".join(str(c).replace("\ufeff", "").split()) for c in df.columns]
    return df


def read_csv_auto(source) -> pd.DataFrame:
    """Lee un CSV desde ruta, URL o archivo subido detectando el separador (',' ';' tab '|')."""
    if hasattr(source, "read"):  # archivo subido (Streamlit / Colab)
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8-sig", errors="replace")
        return clean_columns(pd.read_csv(io.StringIO(raw), sep=None, engine="python"))
    return clean_columns(pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig"))


def to_str(X):
    """Convierte categóricas a texto para que 1 y '1' sean la misma categoría."""
    return pd.DataFrame(X).astype(str).values


# ----------------------------------------------------------------------------
# Inferencia automática de variables
# ----------------------------------------------------------------------------
def infer_feature_types(X: pd.DataFrame, max_cat_unique: int = MAX_CAT_UNIQUE):
    numeric, categorical = [], []
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            categorical.append(col)
        elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= max_cat_unique:
            categorical.append(col)
        else:
            numeric.append(col)
    return numeric, categorical


def suggest_drop_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Sugiere excluir identificadores, constantes y texto libre de alta cardinalidad."""
    n = len(df)
    drop = []
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        nunique = s.nunique(dropna=True)
        name = col.lower()
        is_id_name = name in {"id", "index", "instant"} or name.endswith("_id") or name.startswith("id_")
        if nunique <= 1:
            drop.append(col)
        elif is_id_name:
            drop.append(col)
        elif nunique == n and not pd.api.types.is_float_dtype(s):
            drop.append(col)
        elif s.dtype == object and nunique > 0.5 * n:
            drop.append(col)
    return drop


# ----------------------------------------------------------------------------
# Pipeline (PDF §7.3)
# ----------------------------------------------------------------------------
def build_pipeline(numeric_features, categorical_features, model_name="random_forest") -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("to_str", FunctionTransformer(to_str)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ])
    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000)
    elif model_name == "random_forest":
        # Configuración tomada del notebook de semana 14 del curso
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


# ----------------------------------------------------------------------------
# Entrenamiento + evaluación + artefacto (PDF §7, §8, §14)
# ----------------------------------------------------------------------------
def validate_target(y: pd.Series) -> None:
    n_classes = y.nunique()
    if n_classes < 2:
        raise ValueError("La variable objetivo debe tener al menos 2 clases.")
    if n_classes > MAX_CLASSES:
        raise ValueError(
            f"La variable objetivo tiene {n_classes} valores distintos: parece un problema de "
            "regresión. Esta aplicación implementa la ruta supervisada de CLASIFICACIÓN."
        )


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    classes = list(model.classes_)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        "recall_macro": float(recall_score(y_test, pred, average="macro")),
    }
    proba = model.predict_proba(X_test)
    try:
        if len(classes) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
        else:
            metrics["roc_auc_ovr"] = float(roc_auc_score(y_test, proba, multi_class="ovr", labels=classes))
    except ValueError:
        pass
    return {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=classes).tolist(),
        "classification_report": classification_report(y_test, pred, zero_division=0),
        "labels": [str(c) for c in classes],
    }


def train(
    df: pd.DataFrame,
    target: str,
    drop_columns=None,
    model_name: str = "random_forest",
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
    max_rows: int | None = None,
    dataset_name: str = "dataset.csv",
) -> dict:
    """Entrena el pipeline completo y devuelve el artefacto (dict) listo para joblib."""
    df = clean_columns(df)
    target = " ".join(str(target).split())
    if target not in df.columns:
        raise ValueError(f"La columna objetivo '{target}' no existe en el dataset.")

    drop_columns = [" ".join(str(c).split()) for c in (drop_columns or [])]
    df = df.dropna(subset=[target])
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=random_state)

    y = df[target]
    if y.dtype == object or pd.api.types.is_bool_dtype(y):
        y = y.astype(str)
    validate_target(y)

    features = [c for c in df.columns if c != target and c not in drop_columns]
    if not features:
        raise ValueError("No quedan variables explicativas después de excluir columnas.")
    X = df[features].copy()
    numeric_features, categorical_features = infer_feature_types(X)
    for col in categorical_features:  # categorías como texto (mismo tratamiento que en serving)
        X[col] = X[col].where(X[col].isna(), X[col].astype(str)).astype(object)

    counts = y.value_counts()
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    model = build_pipeline(numeric_features, categorical_features, model_name)
    model.fit(X_train, y_train)
    evaluation = evaluate(model, X_test, y_test)

    category_options = {
        col: sorted(X_train[col].dropna().astype(str).unique().tolist())
        for col in categorical_features
    }
    numeric_stats = {
        col: {
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max()),
            "median": float(X_train[col].median()),
            "integer": bool(pd.api.types.is_integer_dtype(X_train[col])),
        }
        for col in numeric_features
    }
    # Caso de referencia para verificar paridad entre entornos (patrón semana 14)
    ref_input = X_test.iloc[[0]]
    reference = {
        "input": {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                  for k, v in ref_input.iloc[0].to_dict().items()},
        "proba": model.predict_proba(ref_input)[0].tolist(),
        "tolerance": 1e-9,
    }

    return {
        "model": model,
        "features": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "task": "classification",
        "target": target,
        "classes": [c.item() if hasattr(c, "item") else c for c in model.classes_],
        "category_options": category_options,
        "numeric_stats": numeric_stats,
        "dropped_columns": drop_columns,
        "model_name": model_name,
        "metrics": evaluation["metrics"],
        "evaluation": evaluation,
        "dataset": dataset_name,
        "n_rows_train": int(len(X_train)),
        "n_rows_test": int(len(X_test)),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": reference,
    }


# ----------------------------------------------------------------------------
# Inferencia
# ----------------------------------------------------------------------------
def prepare_input(df: pd.DataFrame, artifact: dict):
    """Alinea un DataFrame nuevo con el contrato del artefacto.

    Devuelve (X, faltantes, sobrantes). Las columnas faltantes se crean vacías
    (el imputador del pipeline las completa) y las sobrantes se ignoran.
    """
    df = clean_columns(df)
    features = artifact["features"]
    missing = [c for c in features if c not in df.columns]
    extra = [c for c in df.columns if c not in features]
    X = df.reindex(columns=features)
    for col in artifact["numeric_features"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in artifact["categorical_features"]:
        X[col] = X[col].where(X[col].isna(), X[col].astype(str)).astype(object)
    return X, missing, extra


def predict_df(artifact: dict, df: pd.DataFrame):
    X, missing, extra = prepare_input(df, artifact)
    model = artifact["model"]
    out = df.copy()
    out["prediccion"] = model.predict(X)
    proba = model.predict_proba(X)
    for i, cls in enumerate(model.classes_):
        out[f"prob_{cls}"] = np.round(proba[:, i], 4)
    return out, missing, extra


def check_parity(artifact: dict) -> float:
    """Recalcula el caso de referencia; devuelve la diferencia máxima absoluta."""
    ref = artifact["reference"]
    X = pd.DataFrame([ref["input"]], columns=artifact["features"])
    proba = artifact["model"].predict_proba(X)[0]
    return float(np.max(np.abs(proba - np.array(ref["proba"]))))
