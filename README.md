# Despliegue de Modelos de Machine Learning: Pipeline Supervisado, Contenedorización con Docker y Servicio en Google Cloud Platform

**Asignatura:** Machine Learning II  
**Programa:** Pregrado en Ciencia de Datos  
**Modalidad:** Reto Práctico de Ciclo de Vida Completo de Machine Learning  

---

## 1. Descripción y Propósito del Proyecto

El presente proyecto implementa el ciclo de vida completo de una solución de Machine Learning aplicada, abarcando desde la ingesta de datos y el preprocesamiento automatizado hasta el entrenamiento, evaluación estadística, serialización de artefactos, desarrollo de interfaz interactiva, contenedorización mediante Docker y despliegue en una máquina virtual de infraestructura como servicio (IaaS) en Google Cloud Platform (Compute Engine).

El objetivo principal es garantizar la reproducibilidad estricta del sistema en entornos heterogéneos (Google Colab, entornos locales basados en contenedores y servidores de producción en la nube), asegurando paridad numérica en inferencia y desacoplando la fase de entrenamiento de la fase de servicio.

---

## 2. Conjunto de Datos y Unidad de Observación

- **Fuente:** Repositorio académico oficial del curso (`Customer_Churn_Iran.csv`).
- **Unidad de Análisis:** Cada registro corresponde a un cliente individual de una empresa de telecomunicaciones, evaluado a lo largo de un período de observación de 12 meses.
- **Volumen:** 3.150 observaciones y 14 columnas tabulares sin presencia de valores nulos estructurales.
- **Variable Objetivo (`Target`):** `Churn` (variable binaria, donde `1` representa el abandono del servicio por parte del usuario y `0` representa la retención).
- **Distribución de Clases:** Se presenta un problema de clases desbalanceadas (aproximadamente 15,7% de casos positivos y 84,3% de casos negativos), lo cual exige una estrategia de evaluación especializada.

---

## 3. Ruta Metodológica

El proyecto se desarrolla bajo la **Ruta A (Aprendizaje Supervisado - Clasificación Binaria)** definida en la guía de laboratorio:

- Modelamiento predictivo supervisado con partición estratificada de datos (80% entrenamiento, 20% prueba, `random_state=42`).
- Arquitectura basada en `Pipeline` de `scikit-learn` que integra de forma atómica la imputación, estandarización, codificación categórica y estimador.
- Serialización de artefactos con metadatos completos y caso de referencia para comprobación de paridad numérica.

---

## 4. Definición y Justificación de Variables

A fin de evitar fuga de información (*data leakage*) y asegurar que la interfaz de usuario reciba únicamente información disponible al momento de la toma de decisiones, se aplicó el siguiente esquema de variables. La variable objetivo es `Churn` (1 = abandono, 0 = permanencia), con un desbalance aproximado de 15,7 % positivos / 84,3 % negativos; por eso la partición es estratificada y la evaluación prioriza la clase positiva.

### 4.1 Variables Predictoras Incluidas

| Variable | Tipo | Por qué se incluye | Tratamiento en el pipeline |
|:---|:---|:---|:---|
| `Call Failure` | Numérica discreta (conteo) | Frecuencia de llamadas fallidas; mide problemas de red o insatisfacción y es una señal temprana de abandono. | Mediana + estandarización |
| `Subscription Length` | Numérica (meses) | Antigüedad del cliente; distingue clientes nuevos de clientes con relación establecida. | Mediana + estandarización |
| `Charge Amount` | Numérica discreta (nivel tarifario) | Nivel de cobro o plan; aproxima el valor económico del cliente. | Mediana + estandarización |
| `Seconds of Use` | Numérica continua (segundos) | Intensidad de consumo; un mayor uso sugiere dependencia del servicio. | Mediana + estandarización |
| `Frequency of use` | Numérica discreta (conteo) | Número de llamadas efectuadas; intensidad de la relación con la operadora. | Mediana + estandarización |
| `Frequency of SMS` | Numérica discreta (conteo) | Uso de mensajes de texto; nivel de *engagement* del cliente. | Mediana + estandarización |
| `Distinct Called Numbers` | Numérica discreta (conteo) | Diversidad de contactos; tamaño de la red social del cliente. | Mediana + estandarización |
| `Complains` | Categórica binaria (0/1) | Presencia de queja formal; predictor directo de insatisfacción y churn. | Moda + one-hot |
| `Age Group` | Categórica ordinal (1–5) | Segmento etario; captura heterogeneidad demográfica en la propensión al abandono. | Moda + one-hot (no se impone orden) |
| `Tariff Plan` | Categórica nominal (1 = Contractual, 2 = Prepago) | Tipo de plan; el prepago suele asociarse a mayor rotación. | Moda + one-hot |
| `Status` | Categórica nominal (1, 2) | Estado operativo de la línea; puede reflejar actividad o suspensión del servicio. | Moda + one-hot |

### 4.2 Variables Excluidas

| Variable | Motivo de exclusión |
|:---|:---|
| `Customer Value` | Se calcula *a posteriori* a partir de métricas agregadas de uso. Incluirla generaría fuga de información y no está disponible para un cliente nuevo o en una intervención temprana. |
| `Age` | Presenta una correspondencia **1:1** con `Age Group` (15→1, 25→2, 30→3, 45→4, 55→5). Mantener ambas duplicaría exactamente las mismas columnas one-hot sin aportar información; se conserva `Age Group`, que es el agrupamiento interpretable. |
| `Churn` | Es la variable objetivo; no entra como predictor. |

> No se incluyen identificadores porque el dataset no los contiene. Si existieran columnas como `ID` o `Customer ID`, se excluirían para evitar que el modelo memorice clientes en lugar de aprender patrones.

---

## 5. Preparación de Datos y Arquitectura del Pipeline

Para asegurar que las mismas transformaciones del entrenamiento se repliquen exactamente durante la inferencia, se utilizó un `ColumnTransformer` encapsulado en un `Pipeline`:

```
Entrada tabular (CSV o Formulario)
       │
       ├── Columnas Numéricas ──────> SimpleImputer(median) ─────────> StandardScaler() ───┐
       │                                                                                   │
       └── Columnas Categóricas ────> SimpleImputer(most_frequent) ──> OneHotEncoder() ────┤
                                                                                           │
                                                                   Concatena matriz de features
                                                                                           │
                                                                                           ▼
                                                                             Estimador (Clasificador)
```

- **Manejo de categorías no observadas:** El codificador `OneHotEncoder` utiliza el parámetro `handle_unknown="ignore"`, asignando un vector nulo a categorías no presentes en la fase de entrenamiento, evitando fallas críticas en tiempo de ejecución.
- **Normalización de identificadores de columna:** Los encabezados son procesados para eliminar codificaciones BOM y duplicidad de espacios tipográficos presentes en el archivo fuente.

---

## 6. Selección y Evaluación del Modelo

Se realizó una comparativa empírica entre el modelo base propuesto en la guía (`LogisticRegression`) y un ensamble de árboles (`RandomForestClassifier`, configurado con 300 estimadores, `min_samples_leaf=5`, `class_weight="balanced"` y `random_state=42`).

### 6.1 Selección con Validación Cruzada (5-fold estratificada)

Para evitar el sesgo de elegir "el mejor" modelo sobre el propio conjunto de prueba, la selección se hizo con **validación cruzada estratificada de 5 folds** sobre los datos de entrenamiento, reportando la clase positiva (`Churn=1`):

| Métrica (media ± desv., CV) | Regresión Logística (Base) | Random Forest (Seleccionado) |
|:---|:---:|:---:|
| **F1 (Clase Churn = 1)** | 0.5672 ± 0.0217 | **0.8406 ± 0.0226** |
| **Recall (Clase Churn = 1)** | 0.4404 ± 0.0290 | **0.9414 ± 0.0234** |
| **ROC-AUC** | 0.9354 ± 0.0071 | **0.9824 ± 0.0046** |

El **Random Forest** domina consistentemente en las tres métricas, por lo que fue el modelo seleccionado. El conjunto de prueba se reservó exclusivamente para la evaluación final (sección 6.2), sin participar en la selección.

### 6.2 Matriz de Confusión (Random Forest en Conjunto de Prueba, N = 630)

| Condición Real \ Predicción | Predicho Negativo (0) | Predicho Positivo (1) | Total Real |
|:---|:---:|:---:|:---:|
| **Real Negativo (0)** | 505 | 26 | 531 |
| **Real Positivo (1)** | 4 | 95 | 99 |

### 6.3 Métricas Cuantitativas Comparativas (Conjunto de Prueba)

| Métrica | Regresión Logística (Base) | Random Forest (Seleccionado) |
|:---|:---:|:---:|
| **Accuracy Global** | 0.8968 | **0.9524** |
| **Balanced Accuracy** | 0.7046 | **0.9553** |
| **Recall (Clase Churn = 1)** | 0.4242 | **0.9596** |
| **Precision (Clase Churn = 1)** | 0.8400 | **0.7851** |
| **F1 (Clase Churn = 1)** | 0.5638 | **0.8636** |
| **F1-Score Ponderado** | 0.8821 | **0.9543** |
| **ROC-AUC** | 0.9252 | **0.9844** |
| **Precision-Recall (Average Precision)** | 0.7616 | **0.9353** |

### 6.4 Justificación Técnica de la Elección
Debido a la asimetría en los costos de error asociados al problema de negocio, no identificar a un cliente que efectivamente abandonará el servicio (falso negativo) conlleva la pérdida del valor de vida del cliente (*lifetime value*), mientras que un falso positivo implica únicamente una acción preventiva de contacto. Por consiguiente, el modelo **Random Forest** fue seleccionado por su capacidad de captura (**Recall de 95,96%** frente a 42,42% del modelo lineal) y su **F1 de la clase positiva (0.8636 frente a 0.5638)**. Se reporta el F1 de la clase positiva (no solo el ponderado, dominado por la clase mayoritaria) junto con la precisión-recall (*Average Precision*) para reflejar el desempeño sobre la clase de interés.

---

## 7. Estructura del Repositorio

El proyecto mantiene una organización modular y desacoplada:

```
├── .streamlit/
│   └── config.toml          # Parámetros del servidor Streamlit (headless, maxUploadSize)
├── data/
│   └── datos.csv            # Copia local del dataset Customer_Churn_Iran.csv
├── pdf/
│   └── guia_completa_ml_streamlit_docker_gcp_colab.pdf # Guía académica del taller
├── src/
│   └── build_notebook.py    # Generador automatizado del cuaderno unificado
├── .dockerignore            # Reglas de exclusión para el contexto de Docker
├── .gitignore               # Reglas de exclusión para Git
├── app.py                   # Aplicación interactiva de servicio Streamlit
├── Dockerfile               # Instrucciones de construcción de la imagen de producción
├── ML_Supervisado_Streamlit_Docker_GCP.ipynb # Cuaderno ejecutable en Google Colab
├── ml_core.py               # Módulo central: inferencia de tipos, pipelines, métricas y CV
├── model.joblib             # Artefacto serializado (modelo entrenado + metadatos)
├── README.md                # Documentación técnica institucional
├── requirements.txt         # Dependencias con versiones fijadas (reproducibilidad)
└── train.py                 # Script de entrenamiento por línea de comandos
```

---

## 8. Instrucciones de Reproducción y Ejecución

### 8.1 Ejecución Completa desde Google Colab
1. Cargar el archivo `ML_Supervisado_Streamlit_Docker_GCP.ipynb` en Google Colab.
2. Ejecutar la secuencia a través del comando `Entorno de ejecución` -> `Ejecutar todo`.
3. El cuaderno genera de manera autónoma los scripts modulares, el artefacto entrenado, las pruebas de paridad y descarga el paquete `ml_streamlit_project.zip`.

### 8.2 Ejecución Local en Entorno Virtual

```bash
# 1. Creación e inicialización del entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalación de dependencias del proyecto
pip install --upgrade pip
pip install -r requirements.txt

# 3. Ejecución del script de entrenamiento (reconstrucción de model.joblib)
python train.py

# 4. Despliegue local de la aplicación interactiva
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

Acceder a través del navegador web a `http://localhost:8501`.

> **Nota sobre la pestaña de entrenamiento.** La aplicación no reentrena el modelo al arrancar: siempre sirve `model.joblib`. Incluye, como **extensión opcional**, una pestaña para entrenar con un CSV nuevo (solo si el usuario lo sube y pulsa "Entrenar"). Para no agotar la memoria de la VM, el número de filas está limitado por defecto a 20.000.

---

## 9. Empaquetamiento y Contenedorización con Docker

El sistema se encuentra empaquetado en una imagen ligera basada en `python:3.12-slim`. La aplicación no requiere privilegios elevados y expone el puerto estándar `8501`.

### 9.1 Construcción de la Imagen Docker

```bash
docker build -t ml-streamlit-app .
```

### 9.2 Ejecución y Verificación del Contenedor

```bash
# Ejecución interactiva para verificación de logs de arranque
docker run --rm -p 8501:8501 ml-streamlit-app

# Ejecución en segundo plano con política de reinicio automático
docker run -d \
  --name ml-streamlit \
  --restart unless-stopped \
  -p 8501:8501 \
  ml-streamlit-app

# Inspección de estado y diagnóstico
docker ps
docker logs ml-streamlit
```

### 9.3 Evidencia de Construcción y Ejecución (verificada)

Evidencia registrada el **2026-11-10** en un entorno Linux/WSL con Docker 29.1.3:

| Verificación | Resultado |
|:---|:---|
| `docker build -t ml-streamlit-app .` | `Successfully tagged ml-streamlit-app:latest` |
| `docker images ml-streamlit-app` | Imagen creada (≈245 MB de contenido) |
| `docker ps --filter name=ml-streamlit` | `Up (healthy)`, puerto `0.0.0.0:8501->8501/tcp` |
| `curl http://127.0.0.1:8501/_stcore/health` | `ok` |
| Carga del artefacto dentro del contenedor | `sklearn_version_artifact: 1.9.1`, sin `InconsistentVersionWarning` |
| Paridad numérica del caso de referencia | `parity_diff: 0.0` |
| Predicción de ejemplo dentro del contenedor | Clase `0`, `P(Y=0)=0.5093`, `P(Y=1)=0.4907` |

Los logs completos están en:

- [`docs/evidencias/docker-build.log`](docs/evidencias/docker-build.log)
- [`docs/evidencias/docker-images.log`](docs/evidencias/docker-images.log)
- [`docs/evidencias/docker-run.log`](docs/evidencias/docker-run.log) (incluye `docker ps`, `docker logs` y healthcheck)
- [`docs/evidencias/docker-model-check.log`](docs/evidencias/docker-model-check.log)

Para reproducir la verificación:

```bash
docker build -t ml-streamlit-app .
docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
sleep 10
docker ps
docker logs ml-streamlit
curl http://127.0.0.1:8501/_stcore/health
```

---

## 10. Despliegue en Google Cloud Platform (Compute Engine)

A continuación se detalla el protocolo para desplegar la aplicación en una instancia virtual de Google Cloud:

### 10.1 Creación de la Instancia Virtual
1. Acceder a la consola de administración de Google Cloud Platform (`console.cloud.google.com`).
2. Navegar a **Compute Engine** -> **Instancias de VM**.
3. Seleccionar **Crear instancia** con las siguientes especificaciones:
   - **Sistema operativo:** Ubuntu 24.04 LTS (x86/64).
   - **Tipo de máquina:** `e2-small` o `e2-medium` (recomendado para estabilidad de memoria).
   - **IP Externa:** Asignación efímera o estática habilitada.
   - **Etiqueta de red:** `streamlit-server`.

### 10.2 Configuración del Cortafuegos (Firewall)
Para permitir el tráfico entrante al puerto 8501, ejecutar en Google Cloud Shell o crear la regla vía interfaz de red VPC:

```bash
gcloud compute firewall-rules create allow-streamlit-8501 \
    --allow=tcp:8501 \
    --direction=INGRESS \
    --target-tags=streamlit-server \
    --description="Permitir trafico entrante TCP en puerto 8501 para Streamlit"
```

### 10.3 Instalación, Clonación y Puesta en Marcha en la VM
Conectarse mediante la terminal SSH integrada en la consola de Compute Engine y ejecutar:

```bash
# 1. Actualización e instalación del motor Docker y Git
sudo apt update && sudo apt install -y docker.io git
sudo systemctl enable --now docker

# 2. Clonación del repositorio institucional
git clone https://github.com/julianjimenez4809-ui/MLII_Despliegue_Streamlit_Docker_GCP.git
cd MLII_Despliegue_Streamlit_Docker_GCP

# 3. Construcción y ejecución del contenedor en el servidor
sudo docker build -t ml-streamlit-app .
sudo docker run -d --name ml-streamlit --restart unless-stopped -p 8501:8501 ml-streamlit-app
```

### 10.4 Verificación de Servicio
Abrir un navegador web e ingresar la dirección IP externa de la máquina virtual:

```
http://<IP_EXTERNA_DE_LA_VM>:8501
```

---

## 11. Casos de Prueba y Contratos de Inferencia

### 11.1 Ejemplo de Entrada (Registro de Prueba)
```json
{
  "Call Failure": 2,
  "Complains": "0",
  "Subscription Length": 36,
  "Charge Amount": 0,
  "Seconds of Use": 1260,
  "Frequency of use": 39,
  "Frequency of SMS": 20,
  "Distinct Called Numbers": 22,
  "Age Group": "3",
  "Tariff Plan": "1",
  "Status": "2"
}
```

### 11.2 Salida Generada por el Modelo
- **Clase Predicha:** `0` (Permanencia / No Abandono).
- **Probabilidades Estimadas:** 
  - Probabilidad de Permanencia (`P(Y=0)`): 0.5093
  - Probabilidad de Abandono (`P(Y=1)`): 0.4907

---

## 12. Protocolo de Auditoría y Prueba Cruzada

Para validar la reproducibilidad externa requerida en la sección 31 de la guía, un evaluador independiente debe poder operar el sistema siguiendo exclusivamente los siguientes pasos:

1. Clonar el repositorio mediante la URL pública.
2. Comprobar la integridad de los archivos raíz (`Dockerfile`, `requirements.txt`, `app.py`, `model.joblib`).
3. Ejecutar el comando `sudo docker build -t ml-streamlit-app .` sin alertas de falla de dependencias.
4. Ejecutar el contenedor mediante `sudo docker run -d -p 8501:8501 ml-streamlit-app`.
5. Enviar una petición a través de la pestaña de predicción unitaria o predicción por lotes y verificar la respuesta esperada.

> **Nota sobre permisos.** Los pasos 3 y 4 usan `sudo` porque el usuario que sigue el protocolo en la VM puede no pertenecer aún al grupo `docker`. Quien haya ejecutado `sudo usermod -aG docker $USER` (y reiniciado la sesión SSH) puede omitir `sudo`.

### 12.1 Evidencia de la Prueba Cruzada (PDF §31)

| Campo | Detalle |
|:---|:---|
| **Pareja evaluadora** | *(por completar)* |
| **Fecha** | *(por completar)* |
| **Entorno** | *(SO y versión de Docker)* |
| **Pasos ejecutados** | Clonar → leer README → `docker build` → `docker run` → probar al menos un caso |
| **Resultado** | *(exitoso / con observaciones)* |
| **Errores o instrucciones faltantes** | *(por completar)* |
| **Evidencia** | *(captura / log adjunto)* |

### 12.2 Evidencia del Despliegue en la VM (GCP)

| Campo | Detalle |
|:---|:---|
| **URL / IP externa** | `http://<IP_EXTERNA_DE_LA_VM>:8501` |
| **Fecha del despliegue** | *(por completar)* |
| **Instancia** | *(nombre, zona, tipo e2-small)* |
| **Estado del contenedor** | `docker ps` / `docker logs ml-streamlit` *(por adjuntar)* |
| **Captura de la aplicación** | *(por adjuntar)* |

---

## 13. Desaprovisionamiento y Control de Costos

Al finalizar la demostración evaluativa o sesión de auditoría, es indispensable detener o desaprovisionar los recursos de cómputo para evitar facturación recurrente en Google Cloud:

```bash
# Detención de la máquina virtual vía gcloud CLI
gcloud compute instances stop <NOMBRE_DE_LA_VM> --zone=<ZONA>

# O eliminación definitiva de la instancia y sus discos asociados
gcloud compute instances delete <NOMBRE_DE_LA_VM> --zone=<ZONA>
```

---

## 14. Autores

- **Julián Jiménez**  
  *Programa de Pregrado en Ciencia de Datos*  
  *Repositorio Académico:* [julianjimenez4809-ui](https://github.com/julianjimenez4809-ui)

- **Tomás Rincón**  
  *Programa de Pregrado en Ciencia de Datos*
