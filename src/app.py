"""app.py - Aplicación Streamlit (PDF §13-14 + extensión de entrenamiento con CSV nuevo).

Pestañas:
  1. Entrenar con un CSV nuevo (cualquier dataset de clasificación).
  2. Predecir un caso (formulario generado desde el artefacto).
  3. Predecir por lotes (CSV) y descargar resultados.
"""
from __future__ import annotations

import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import ml_core

MODEL_PATH = Path("model.joblib")

st.set_page_config(page_title="ML App - Clasificación supervisada", layout="wide")


@st.cache_resource
def load_default_artifact():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data(show_spinner=False)
def read_uploaded(data: bytes) -> pd.DataFrame:
    return ml_core.read_csv_auto(io.BytesIO(data))


if "artifact" not in st.session_state:
    st.session_state["artifact"] = load_default_artifact()
artifact = st.session_state["artifact"]

# ----------------------------------------------------------------------------
# Barra lateral: modelo activo
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Modelo activo")
    if artifact is None:
        st.warning("No hay modelo cargado. Entrene uno en la pestaña 1.")
    else:
        st.write(f"**Dataset:** {artifact.get('dataset', '-')}")
        st.write(f"**Objetivo:** `{artifact['target']}`")
        st.write(f"**Modelo:** {ml_core.MODELS.get(artifact['model_name'], artifact['model_name'])}")
        st.write(f"**Clases:** {', '.join(map(str, artifact['classes']))}")
        st.write(f"**Variables:** {len(artifact['features'])}")
        st.json({k: round(v, 4) for k, v in artifact["metrics"].items()}, expanded=False)
        st.download_button(
            "Descargar model.joblib",
            data=(lambda b: (joblib.dump(artifact, b), b.getvalue())[1])(io.BytesIO()),
            file_name="model.joblib",
        )
        if MODEL_PATH.exists() and st.button("Restaurar modelo por defecto"):
            st.session_state["artifact"] = load_default_artifact()
            st.rerun()

st.title("Aplicación de Machine Learning supervisado")
st.caption("Pipeline scikit-learn (preprocesamiento + modelo) · Streamlit · Docker · GCP")

tab_train, tab_single, tab_batch = st.tabs(
    ["1 · Entrenar con nuevo CSV", "2 · Predecir un caso", "3 · Predecir por lotes (CSV)"]
)

# ----------------------------------------------------------------------------
# 1. Entrenar
# ----------------------------------------------------------------------------
with tab_train:
    st.subheader("Entrenar un modelo con cualquier CSV")
    st.write("Seleccione un archivo CSV (separador `,` o `;` detectado automáticamente).")
    up = st.file_uploader("Explorar archivos", type=["csv"], key="train_csv")
    if up is not None:
        try:
            df = read_uploaded(up.getvalue())
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo leer el CSV: {exc}")
            st.stop()

        c1, c2, c3 = st.columns(3)
        c1.metric("Filas", f"{df.shape[0]:,}")
        c2.metric("Columnas", df.shape[1])
        c3.metric("Celdas nulas", int(df.isna().sum().sum()))
        st.dataframe(df.head(20))
        with st.expander("Tipos y nulos por columna"):
            st.dataframe(pd.DataFrame({
                "tipo": df.dtypes.astype(str),
                "nulos": df.isna().sum(),
                "valores_distintos": df.nunique(),
            }))

        cols = df.columns.tolist()
        target = st.selectbox("Columna objetivo (y)", cols, index=len(cols) - 1)
        suggested = ml_core.suggest_drop_columns(df, target)
        drop = st.multiselect(
            "Columnas a excluir (identificadores, fuga de información...)",
            [c for c in cols if c != target], default=suggested,
        )
        cA, cB, cC = st.columns(3)
        model_name = cA.selectbox("Modelo", list(ml_core.MODELS), format_func=ml_core.MODELS.get, index=1)
        test_size = cB.slider("Proporción de prueba", 0.1, 0.4, 0.2, 0.05)
        max_rows = cC.number_input("Máximo de filas (0 = todas)", min_value=0, value=0, step=1000)

        st.write("Distribución de la variable objetivo:")
        st.dataframe(df[target].value_counts(dropna=False).rename("conteo").to_frame().T)

        if st.button("Entrenar", type="primary"):
            with st.spinner("Entrenando..."):
                try:
                    new_art = ml_core.train(
                        df, target=target, drop_columns=drop, model_name=model_name,
                        test_size=test_size, max_rows=int(max_rows) or None, dataset_name=up.name,
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                    st.stop()
            st.session_state["artifact"] = new_art
            artifact = new_art
            st.success("Modelo entrenado y activado. Ya puede usar las pestañas 2 y 3.")
            ev = new_art["evaluation"]
            st.write("**Métricas en el conjunto de prueba**")
            st.dataframe(pd.Series(new_art["metrics"]).round(4).to_frame("valor").T)
            st.write("**Matriz de confusión** (filas = real, columnas = predicho)")
            st.dataframe(pd.DataFrame(ev["confusion_matrix"], index=ev["labels"], columns=ev["labels"]))
            st.code(ev["classification_report"])
            st.info(f"Numéricas: {new_art['numeric_features']}\n\nCategóricas: {new_art['categorical_features']}")
            if st.button("Actualizar interfaz"):
                st.rerun()

if artifact is None:
    with tab_single:
        st.info("Primero entrene un modelo.")
    with tab_batch:
        st.info("Primero entrene un modelo.")
    st.stop()

# ----------------------------------------------------------------------------
# 2. Un caso (PDF §13-14)
# ----------------------------------------------------------------------------
with tab_single:
    st.subheader(f"Predecir `{artifact['target']}` para un nuevo caso")
    values = {}
    with st.form("single"):
        grid = st.columns(3)
        i = 0
        for feature in artifact["numeric_features"]:
            stats = artifact["numeric_stats"][feature]
            with grid[i % 3]:
                if stats["integer"]:
                    values[feature] = st.number_input(feature, value=int(round(stats["median"])), step=1,
                                                      help=f"rango entrenamiento: {stats['min']:g} – {stats['max']:g}")
                else:
                    values[feature] = st.number_input(feature, value=float(stats["median"]),
                                                      help=f"rango entrenamiento: {stats['min']:g} – {stats['max']:g}")
            i += 1
        for feature in artifact["categorical_features"]:
            with grid[i % 3]:
                values[feature] = st.selectbox(feature, artifact["category_options"][feature])
            i += 1
        submitted = st.form_submit_button("Ejecutar modelo", type="primary")
    if submitted:
        X_new = pd.DataFrame([values], columns=artifact["features"])
        out, _, _ = ml_core.predict_df(artifact, X_new)
        st.success(f"Clase predicha: **{out['prediccion'].iloc[0]}**")
        probs = out[[c for c in out.columns if c.startswith("prob_")]].T
        probs.columns = ["probabilidad"]
        st.bar_chart(probs)
        st.dataframe(probs)

# ----------------------------------------------------------------------------
# 3. Lotes
# ----------------------------------------------------------------------------
with tab_batch:
    st.subheader("Predicción por lotes")
    st.write("Suba un CSV con las columnas del modelo activo (la columna objetivo es opcional):")
    st.code(", ".join(artifact["features"]))
    template = pd.DataFrame([{f: artifact["reference"]["input"].get(f) for f in artifact["features"]}])
    st.download_button("Descargar plantilla CSV", template.to_csv(index=False).encode(), "plantilla.csv")
    up_b = st.file_uploader("Explorar archivos", type=["csv"], key="batch_csv")
    if up_b is not None:
        df_b = read_uploaded(up_b.getvalue())
        out, missing, extra = ml_core.predict_df(artifact, df_b)
        if missing:
            st.warning(f"Columnas faltantes (se imputan): {missing}")
        if extra:
            st.info(f"Columnas ignoradas: {extra}")
        st.dataframe(out)
        st.write("Distribución de predicciones:")
        st.bar_chart(out["prediccion"].astype(str).value_counts())
        if artifact["target"] in df_b.columns:
            acc = (out["prediccion"].astype(str) == df_b[artifact["target"]].astype(str)).mean()
            st.metric("Accuracy contra la columna objetivo incluida", f"{acc:.3f}")
        st.download_button("Descargar predicciones", out.to_csv(index=False).encode(), "predicciones.csv")
