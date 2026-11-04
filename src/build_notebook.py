"""Genera el notebook único ML_Supervisado_Streamlit_Docker_GCP.ipynb.

Los archivos fuente (ml_core.py, train.py, app.py) se mantienen en src/ y se
incrustan en el notebook mediante celdas %%writefile, de modo que el notebook
sea autocontenido y reproducible en Google Colab.
"""
import json
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC.parent / "ML_Supervisado_Streamlit_Docker_GCP.ipynb"
P = "/content/ml_streamlit_project"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "id": f"c{len(cells):02d}", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "id": f"c{len(cells):02d}", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


def writefile(name):
    code(f"%%writefile {P}/{name}\n" + (SRC / name).read_text())


md("""
# Reto completo: del dataset al despliegue de Machine Learning
## Ruta A – Aprendizaje supervisado (clasificación)
**Google Colab + datos propios + Streamlit + GitHub + Docker + Google Cloud Compute Engine**

**Autores (pareja):** _Nombre 1_ · _Nombre 2_

Este notebook es **el único entregable de desarrollo**: al ejecutarlo de principio a fin
(`Entorno de ejecución → Ejecutar todo`) genera todo el proyecto en `/content/ml_streamlit_project`:

| Archivo | Rol |
|---|---|
| `data/datos.csv` | Dataset (Customer Churn Iran, repo del curso) |
| `ml_core.py` | Lógica común: inferencia de variables, Pipeline, evaluación, artefacto, predicción |
| `train.py` | Entrenamiento reproducible → `model.joblib` |
| `model.joblib` | Artefacto: pipeline + metadatos (features, categorías, rangos, métricas) |
| `app.py` | Streamlit: **entrenar con cualquier CSV**, predecir un caso, predecir por lotes |
| `requirements.txt`, `Dockerfile`, `.dockerignore`, `.streamlit/config.toml` | Empaquetado |
| `README.md` | Instrucciones completas de reproducción y despliegue |

Flujo (PDF §3): dataset → exploración → `train.py` → entrenar/evaluar → `model.joblib` → `app.py` →
probar Streamlit en Colab → archivos Docker → GitHub → VM GCP → Docker → `git clone` → `docker build` →
`docker run` → firewall → IP pública → prueba cruzada.
""")

md("## 4. Preparación del proyecto en Google Colab\n### Paso 4.1 – Instalar dependencias")
code("!pip -q install streamlit scikit-learn pandas numpy joblib openpyxl")

md("### Paso 4.2 – Crear la carpeta del proyecto")
code(f"""
from pathlib import Path
import os, sys

PROJECT = Path("{P}")
PROJECT.mkdir(exist_ok=True)
(PROJECT / "data").mkdir(exist_ok=True)
(PROJECT / ".streamlit").mkdir(exist_ok=True)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT))
print(PROJECT)
""")

md("""
### Paso 4.3 – Recordatorio sobre persistencia
El almacenamiento de Colab es **temporal**: un reinicio del runtime puede borrar los archivos.
Al final del notebook se genera `ml_streamlit_project.zip` para descargarlo y subirlo a GitHub.
""")

md("""
## Configuración del experimento
Cambiando **solo esta celda** el proyecto completo se reentrena con otro CSV de clasificación
(por ejemplo `Default_Clientes.csv` del repo, objetivo `Default`, que usa `;` como separador).
""")
code("""
DATA_URL = ("https://raw.githubusercontent.com/wilmerpineda/machine_learning_ii_course_public/"
            "main/datasets/public/Customer_Churn_Iran.csv")
TARGET = "Churn"
EXTRA_DROP = ["Customer Value"]   # además de las sugeridas automáticamente (IDs, constantes)
MODEL_NAME = "random_forest"      # "logistic_regression" o "random_forest"
USE_UPLOAD = False                # True -> seleccionar un CSV/Excel propio desde el computador
RANDOM_STATE = 42
""")

md("""
## Módulo común `ml_core.py`
Contiene el Pipeline del PDF §7.3 (imputación + escalado / imputación + OneHot + modelo),
la inferencia automática de variables numéricas/categóricas, la evaluación del PDF §8.1 y la
construcción del artefacto (PDF §8.2 y §14). Lo usan **tanto `train.py` como `app.py`**, por lo que
las transformaciones de entrenamiento y de serving son idénticas.
""")
writefile("ml_core.py")

md("## 5. Cargar los datos\nOpción B (URL pública, por defecto) u Opción A (subir archivo propio).")
code("""
import pandas as pd
import ml_core

if USE_UPLOAD:
    from google.colab import files
    uploaded = files.upload()
    filename = next(iter(uploaded))
    print("Archivo cargado:", filename)
    if filename.lower().endswith(".csv"):
        df = ml_core.read_csv_auto(filename)
    elif filename.lower().endswith((".xlsx", ".xls")):
        df = ml_core.clean_columns(pd.read_excel(filename))
    else:
        raise ValueError("Use un archivo CSV o Excel.")
else:
    df = ml_core.read_csv_auto(DATA_URL)

df.head()
""")

md("""
## 6. Revisión mínima del dataset
- **¿Qué representa una fila?** Un cliente de una compañía de telecomunicaciones iraní, con su uso agregado de 12 meses.
- **Objetivo:** predecir `Churn` (1 = el cliente abandona).
- **¿Qué variables pueden entrar?** Las de uso y perfil. `Customer Value` se excluye porque se calcula a partir del uso (riesgo de fuga) y no está disponible para un cliente nuevo.
- Los nombres con doble espacio (`Call  Failure`) se normalizan con `ml_core.clean_columns`.
""")
code("""
print(df.shape)
display(df.head())
display(df.dtypes)
display(df.isna().sum().sort_values(ascending=False).head(15))
display(df.describe(include="all").T)
print("Duplicados:", df.duplicated().sum())
display(df[TARGET].value_counts(normalize=True).rename("proporción"))
""")

md("## 7. Ruta A – Construcción supervisada\n### Paso 7.1 – Definir objetivo y variables (inferencia automática)")
code("""
drop_columns = ml_core.suggest_drop_columns(df, TARGET)
drop_columns += [c for c in EXTRA_DROP if c in df.columns and c not in drop_columns]

features = [c for c in df.columns if c not in drop_columns + [TARGET]]
X = df[features].copy()
y = df[TARGET].copy()
numeric_features, categorical_features = ml_core.infer_feature_types(X)

print("Excluidas:", drop_columns)
print("Numéricas:", numeric_features)
print("Categóricas:", categorical_features)
""")

md("### Paso 7.2 – Separar entrenamiento y prueba")
code("""
from sklearn.model_selection import train_test_split

for col in categorical_features:
    X[col] = X[col].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(X_train.shape, X_test.shape)
""")

md("""
### Paso 7.3 – Preprocesamiento + modelo (Pipeline)
Modelo base del PDF: regresión logística. Se compara con Random Forest (configuración del notebook
de semana 14 del curso: 300 árboles, `min_samples_leaf=5`, `class_weight="balanced"`).
""")
code("""
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score

results = {}
for name in ml_core.MODELS:
    model = ml_core.build_pipeline(numeric_features, categorical_features, name)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "f1_weighted": f1_score(y_test, pred, average="weighted"),
        "recall_clase_1": recall_score(y_test, pred, pos_label=model.classes_[-1]),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if len(model.classes_) == 2 else None,
    }
    print(f"===== {ml_core.MODELS[name]} =====")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))

pd.DataFrame(results).T.round(4)
""")

md("""
## 8. Evaluación y elección del modelo
Las clases están desbalanceadas (≈16 % de churn), por eso *accuracy* no basta (PDF §8.1).
Elegimos el modelo con mejor **recall de la clase churn** y **F1 ponderado**: no detectar un cliente que
se va (falso negativo) cuesta más que contactar a uno que se habría quedado. ROC-AUC sirve de apoyo
porque no depende del umbral.
""")
code("""
best = max(results, key=lambda k: (results[k]["recall_clase_1"], results[k]["f1_weighted"]))
print("Modelo elegido:", ml_core.MODELS[best])
if best != MODEL_NAME:
    print(f"Nota: MODEL_NAME está configurado como '{MODEL_NAME}'; train.py usará el elegido -> '{best}'")
MODEL_NAME = best
""")

md("### Paso 8.2 – Guardar artefacto (modelo + metadatos + `category_options`)")
code("""
import joblib

artifact = ml_core.train(df, TARGET, drop_columns, MODEL_NAME, random_state=RANDOM_STATE,
                         dataset_name="datos.csv")
joblib.dump(artifact, PROJECT / "model.joblib")
print({k: round(v, 4) for k, v in artifact["metrics"].items()})
print("Claves del artefacto:", list(artifact))
""")

md("## 11–12. Script reproducible `train.py` y dataset dentro del proyecto")
code("""
df.to_csv(PROJECT / "data" / "datos.csv", index=False)
print("Dataset guardado en", PROJECT / "data" / "datos.csv", df.shape)
""")
writefile("train.py")

md("Reentrenar desde cero con el script (PDF §16.1): debe regenerar `model.joblib` con las mismas métricas.")
code("""
drop_args = " ".join(f'"{c}"' for c in drop_columns)
!cd {PROJECT} && python train.py --target "{TARGET}" --model {MODEL_NAME} --drop {drop_args}
!ls -la {PROJECT}
""")

md("""
## 13–14. Aplicación Streamlit `app.py`
Tres pestañas:
1. **Entrenar con nuevo CSV**: botón para seleccionar cualquier CSV, elegir la columna objetivo, las columnas a excluir y el modelo. Se entrena con el **mismo pipeline** de `ml_core.py`, se muestran las métricas y el modelo queda activo (y se puede descargar).
2. **Predecir un caso**: el formulario se genera desde el artefacto (`number_input` con mediana/rango y `selectbox` con las categorías vistas en entrenamiento, PDF §14).
3. **Predecir por lotes**: se sube un CSV, se validan las columnas (las faltantes se imputan y las sobrantes se ignoran) y se descarga el resultado.

Al arrancar, la app carga `model.joblib` **sin reentrenar** (PDF §34). Entrenar desde la app es una extensión sobre el PDF.
""")
writefile("app.py")

code(f"""
%%writefile {P}/.streamlit/config.toml
[server]
headless = true
maxUploadSize = 200

[browser]
gatherUsageStats = false
""")

md("## 15. Probar Streamlit en Google Colab")
code(f"""
# Equivalente a: !streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > /content/streamlit.log 2>&1 &
import subprocess, time
subprocess.run(["pkill", "-f", "streamlit run"])
log = open("/content/streamlit.log", "w")
st_proc = subprocess.Popen(
    ["streamlit", "run", "{P}/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
    cwd="{P}", stdout=log, stderr=subprocess.STDOUT,
)
time.sleep(10)
print(open("/content/streamlit.log").read())
!curl -s -o /dev/null -w "Health check HTTP %{{http_code}}\\n" http://localhost:8501/_stcore/health
""")
md("""
Túnel temporal para verlo desde el navegador (solo para pruebas, **no** es el despliegue final).
La contraseña que pide localtunnel es la IP pública de Colab que imprime la primera línea.
Si la red bloquea el túnel, basta con que el log anterior muestre que Streamlit inició.
Detenga la celda (■) para cerrar el túnel.
""")
code("""
!curl -s https://loca.lt/mytunnelpassword; echo
!npx --yes localtunnel --port 8501
""")

md("## 16. Pruebas mínimas antes de Docker")
code("""
import numpy as np

loaded = joblib.load(PROJECT / "model.joblib")

# 1) Paridad: el artefacto recargado reproduce la predicción de referencia
diff = ml_core.check_parity(loaded)
assert diff <= 1e-9, diff
print("Paridad verificada, diferencia máxima:", diff)

# 2) Las columnas de entrada coinciden con las de entrenamiento
assert loaded["features"] == [c for c in df.columns if c not in drop_columns + [TARGET]]

# 3) Tres casos distintos + valores extremos + categoría inválida + columna faltante/sobrante
casos = df.drop(columns=[TARGET]).sample(3, random_state=1).copy()
extremo = casos.iloc[[0]].copy()
for c in loaded["numeric_features"]:
    extremo[c] = loaded["numeric_stats"][c]["max"] * 10
if loaded["categorical_features"]:
    extremo[loaded["categorical_features"][0]] = "categoria_nunca_vista"
lote = pd.concat([casos, extremo]).drop(columns=[loaded["features"][0]])
lote["columna_extra"] = 1
lote.iloc[1, 0] = np.nan
out, missing, extra = ml_core.predict_df(loaded, lote)
print("Faltantes:", missing, "| Ignoradas:", extra)
display(out.iloc[:, -(len(loaded["classes"]) + 1):])

# 4) Entrenar con otro dataset del repo (lo mismo que hace la pestaña 1 de la app)
other = ml_core.read_csv_auto("https://raw.githubusercontent.com/wilmerpineda/machine_learning_ii_course_public/"
                              "main/datasets/public/Default_Clientes.csv")
art2 = ml_core.train(other, "Default", ml_core.suggest_drop_columns(other, "Default"), "logistic_regression")
print("Default_Clientes ->", {k: round(v, 4) for k, v in art2["metrics"].items()})

# 5) Ejemplo de entrada/salida para el README
ejemplo = pd.DataFrame([loaded["reference"]["input"]])
EJEMPLO_OUT, _, _ = ml_core.predict_df(loaded, ejemplo)
display(EJEMPLO_OUT)
""")

md("""
## 17. Archivos para Docker
`requirements.txt` se escribe con **las versiones exactas instaladas en este Colab**: el `model.joblib`
debe cargarse en Docker con la misma versión de scikit-learn con la que se entrenó.
""")
code("""
from importlib.metadata import version
pkgs = ["streamlit", "pandas", "numpy", "scikit-learn", "joblib", "openpyxl"]
reqs = "\\n".join(f"{p}=={version(p)}" for p in pkgs) + "\\n"
(PROJECT / "requirements.txt").write_text(reqs)
print(reqs)
print("Python de Colab:", sys.version.split()[0])
""")
code(f"""
%%writefile {P}/.dockerignore
.git
.venv
__pycache__
*.pyc
.ipynb_checkpoints
.env
*.log
""")
code("""
# Dockerfile del PDF §17. La versión de Python de la imagen base se iguala a la de Colab
# (PDF: python:3.12-slim) para que model.joblib se cargue con las mismas librerías.
PY = f"{sys.version_info.major}.{sys.version_info.minor}"
dockerfile = f'''FROM python:{PY}-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \\\\
     "--server.address=0.0.0.0", \\\\
     "--server.port=8501"]
'''
(PROJECT / "Dockerfile").write_text(dockerfile)
print(dockerfile)
""")

md("""
## 18. ¿Qué significa cada instrucción Docker?
| Instrucción | Significado |
|---|---|
| `FROM python:3.X-slim` | Imagen base pequeña con Python (misma versión que el Colab donde se entrenó; hoy 3.12) |
| `WORKDIR /app` | Directorio de trabajo dentro de la imagen |
| `COPY requirements.txt .` | Copia primero las dependencias para aprovechar la caché de capas |
| `RUN pip install ...` | Instala las librerías dentro de la imagen |
| `COPY . .` | Copia el proyecto (app.py, ml_core.py, model.joblib, data/, .streamlit/) |
| `EXPOSE 8501` | Documenta el puerto de Streamlit |
| `CMD ...` | Proceso que se ejecuta al iniciar el contenedor |

El Dockerfile se escribe en Colab, pero la imagen **se construye en la VM** de GCP.
""")

md("## 20. README completo")
code('''
m = loaded["metrics"]
metricas_md = "\\n".join(f"| {k} | {v:.4f} |" for k, v in m.items())
import json
ej_in = json.dumps(loaded["reference"]["input"], ensure_ascii=False, indent=2, default=str)
probs = {c: float(EJEMPLO_OUT[c].iloc[0]) for c in EJEMPLO_OUT.columns if c.startswith("prob_")}
ej_out = f"Clase predicha: {EJEMPLO_OUT['prediccion'].iloc[0]} | Probabilidades: {probs}"

readme = f"""# Reto ML supervisado: predicción de abandono (churn) con Streamlit + Docker + GCP

**Autores:** _Nombre 1_ · _Nombre 2_

## Objetivo
Predecir si un cliente de telecomunicaciones abandonará la compañía (`{TARGET}` = 1) y servir el modelo en una
aplicación Streamlit empaquetada con Docker y desplegada en una VM de Google Cloud. La app también permite
**entrenar un modelo nuevo con cualquier CSV de clasificación** subido desde el navegador.

## Dataset y unidad de observación
- Fuente: `Customer_Churn_Iran.csv`, del repositorio del curso
  (https://github.com/wilmerpineda/machine_learning_ii_course_public/tree/main/datasets/public), copiado en `data/datos.csv`.
- Cada fila es **un cliente** con su uso agregado de 12 meses. {df.shape[0]} filas y {df.shape[1]} columnas, sin valores nulos.
- Si `data/datos.csv` no existe, `train.py` lo descarga desde la URL pública.

## Ruta elegida
**Ruta A – Aprendizaje supervisado**, clasificación binaria.

## Variables utilizadas
- Numéricas: {", ".join(loaded["numeric_features"])}
- Categóricas: {", ".join(loaded["categorical_features"])}
- Excluidas: {", ".join(loaded["dropped_columns"])} (`Customer Value` se deriva del uso: riesgo de fuga de información).
- El tipo de cada variable se infiere automáticamente: texto, o enteros con 10 o menos valores distintos, cuentan como categóricas.

## Preparación de datos
Todo vive dentro de un `Pipeline` de scikit-learn (`ml_core.build_pipeline`):
- Numéricas: `SimpleImputer(median)` → `StandardScaler`.
- Categóricas: `SimpleImputer(most_frequent)` → texto → `OneHotEncoder(handle_unknown="ignore")`.
- Nombres de columna normalizados (espacios dobles). Partición 80/20 estratificada, `random_state=42`.

## Modelo y justificación
Se compararon regresión logística (modelo base del PDF) y Random Forest (300 árboles, `min_samples_leaf=5`,
`class_weight="balanced"`, configuración del notebook de semana 14). Modelo elegido: **{ml_core.MODELS[loaded["model_name"]]}**,
por su mayor recall de la clase churn y su mayor F1 ponderado.

## Métricas (conjunto de prueba, {loaded["n_rows_test"]} clientes)
| Métrica | Valor |
|---|---|
{metricas_md}

Las clases están desbalanceadas, así que se prioriza recall y F1 sobre accuracy: no detectar a un cliente que se va es el error más costoso.

## Interpretación de la salida
La app devuelve la **clase predicha** (0 = permanece, 1 = abandona) y la **probabilidad de cada clase**.
Una probabilidad alta de 1 indica que conviene una acción de retención. Es una demostración académica: no se debe
usar para decisiones reales sin validación externa.

## Estructura
```
data/datos.csv  ml_core.py  train.py  model.joblib  app.py
requirements.txt  Dockerfile  .dockerignore  .streamlit/config.toml  README.md
```

## Cómo ejecutar train.py
```bash
pip install -r requirements.txt
python train.py                                   # dataset por defecto -> model.joblib
python train.py --data otro.csv --target clase    # cualquier CSV de clasificación
```

## Cómo ejecutar Streamlit
```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
```
Pestañas: **1) Entrenar con nuevo CSV**, **2) Predecir un caso**, **3) Predecir por lotes (CSV)**.

## Cómo construir la imagen Docker
```bash
docker build -t ml-streamlit-app .
docker images
```

## Cómo ejecutar el contenedor
```bash
docker run --rm -p 8501:8501 ml-streamlit-app          # prueba interactiva (Ctrl+C para salir)
docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
docker ps
docker logs ml-streamlit
```
Abrir `http://IP_EXTERNA_DE_LA_VM:8501`.

## Despliegue en Google Cloud (Compute Engine)
1. Consola de GCP → Compute Engine (habilitar la API si la consola lo pide) → Crear instancia: **Ubuntu 24.04 LTS**, tamaño pequeño (por ejemplo e2-small), IP externa, etiqueta de red `streamlit-server`.
2. Botón **SSH** y luego:
```bash
sudo apt update && sudo apt install -y docker.io git
sudo systemctl enable --now docker
git clone <URL_DEL_REPOSITORIO> && cd <NOMBRE_DEL_REPOSITORIO>
ls -la                       # Dockerfile, app.py, ml_core.py, model.joblib, requirements.txt
sudo docker build -t ml-streamlit-app .
sudo docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
```
3. Firewall (Cloud Shell o VPC → Firewall):
```bash
gcloud compute firewall-rules create allow-streamlit-8501 --allow=tcp:8501 --direction=INGRESS --target-tags=streamlit-server
```
4. Abrir `http://IP_EXTERNA:8501`.
5. **Al terminar, detener o eliminar la VM** y la regla de firewall para no generar costos.

## Actualizar la aplicación desplegada
```bash
git pull
sudo docker stop ml-streamlit && sudo docker rm ml-streamlit
sudo docker build -t ml-streamlit-app .
sudo docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
```

## Diagnóstico
| Situación | Revisión |
|---|---|
| train.py falla | ruta del dataset, nombre de la columna objetivo, dependencias |
| Contenedor se detiene | `docker ps -a`, `docker logs ml-streamlit` |
| Puerto no responde | `-p 8501:8501` y `--server.address=0.0.0.0` |
| Funciona en la VM pero no desde fuera | regla de firewall, etiqueta `streamlit-server`, IP externa |
| Error al cargar model.joblib | versiones de `requirements.txt` iguales a las del entrenamiento |

## Ejemplo de entrada y salida
Entrada:
```json
{ej_in}
```
Salida: `{ej_out}`

## Reproducción completa desde Colab
Abrir `ML_Supervisado_Streamlit_Docker_GCP.ipynb` en Google Colab → *Ejecutar todo*. Se regenera todo el proyecto.
"""
(PROJECT / "README.md").write_text(readme, encoding="utf-8")
print(readme[:1500])
''')

md("## 21. Empaquetar y descargar el proyecto (para subirlo a GitHub)")
code("""
!cd /content && rm -f ml_streamlit_project.zip && zip -qr ml_streamlit_project.zip ml_streamlit_project -x "*/__pycache__/*"
!ls -la /content/ml_streamlit_project /content/ml_streamlit_project.zip
try:
    from google.colab import files
    files.download("/content/ml_streamlit_project.zip")
except ImportError:
    print("Fuera de Colab: el zip está en /content/ml_streamlit_project.zip")
""")

md("""
## 22–31. Despliegue en GCP y prueba cruzada (pasos manuales)
1. Crear un repositorio en GitHub y subir el **contenido descomprimido** del zip, incluido este notebook. No subir credenciales.
2. Seguir la sección *Despliegue en Google Cloud* del `README.md` (VM Ubuntu 24.04 → Docker → `git clone` → `docker build` → `docker run` → firewall → `http://IP:8501`).
3. **Prueba cruzada (PDF §31):** otra pareja recibe solo la URL del repositorio, sigue únicamente el README, construye, ejecuta, prueba un caso y registra los problemas que encuentre.
4. Detener o eliminar la VM al finalizar.

## 34. Checklist final
- [ ] Entendemos qué representa una fila · [ ] Ruta supervisada definida · [ ] Variables justificadas
- [ ] Preprocesamiento dentro del pipeline · [ ] `train.py` corre de principio a fin · [ ] `model.joblib` se genera
- [ ] `app.py` no reentrena al arrancar · [ ] Probamos Streamlit en Colab · [ ] requirements/Dockerfile/.dockerignore
- [ ] Repositorio completo · [ ] VM con SSH · [ ] Docker instalado · [ ] `docker build` OK · [ ] `docker run` activo
- [ ] `docker logs` sin errores · [ ] Firewall · [ ] App abre con la IP externa · [ ] Prueba cruzada · [ ] VM detenida/eliminada
""")

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {"colab": {"provenance": []},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"},
                 "language_info": {"name": "python"}},
    "cells": cells,
}
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook generado:", OUT, "celdas:", len(cells))
