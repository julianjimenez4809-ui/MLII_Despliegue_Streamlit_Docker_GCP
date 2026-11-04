# Despliegue de Modelos de Machine Learning: Pipeline Supervisado, Contenedorización con Docker y Servicio en Google Cloud Platform

**Asignatura:** Machine Learning II  
**Programa:** Maestría en Inteligencia Artificial / Maestría en Ciencia de Datos  
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

A fin de evitar fuga de información (*data leakage*) y asegurar que la interfaz de usuario reciba únicamente información disponible al momento de la toma de decisiones, se aplicó el siguiente esquema de variables:

### 4.1 Variables Predictoras Incluidas
- **Numéricas Continuas y Discretas:**
  - `Call Failure`: Frecuencia acumulada de llamadas fallidas.
  - `Subscription Length`: Duración de la suscripción del cliente (en meses).
  - `Charge Amount`: Nivel tarifario o categoría de cobro.
  - `Seconds of Use`: Segundos totales de consumo telefónico.
  - `Frequency of use`: Número total de llamadas efectuadas.
  - `Frequency of SMS`: Número total de mensajes de texto enviados.
  - `Distinct Called Numbers`: Cantidad de números telefónicos distintos contactados.
- **Categóricas y Discretas de Baja Cardinalidad:**
  - `Complains`: Presencia de quejas o reclamos formales (0 = No, 1 = Sí).
  - `Age Group`: Grupo etario del usuario.
  - `Tariff Plan`: Modalidad de plan contratado (1 = Contractual, 2 = Prepago).
  - `Status`: Estado operativo de la línea telefónica.
  - `Age`: Rango etario reportado.

### 4.2 Variables Excluidas
- `Customer Value`: Variable compuesta calculada *a posteriori* a partir de métricas agregadas del uso. Su inclusión en el entrenamiento generaría fuga de datos, ya que no representa una variable observable de manera inmediata en clientes nuevos o en escenarios de intervención temprana.

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

### 6.1 Matriz de Confusión (Random Forest en Conjunto de Prueba, N = 630)

| Condición Real \ Predicción | Predicho Negativo (0) | Predicho Positivo (1) | Total Real |
|:---|:---:|:---:|:---:|
| **Real Negativo (0)** | 502 | 29 | 531 |
| **Real Positivo (1)** | 4 | 95 | 99 |

### 6.2 Métricas Cuantitativas Comparativas

| Métrica | Regresión Logística (Base) | Random Forest (Seleccionado) |
|:---|:---:|:---:|
| **Accuracy Global** | 0.8984 | **0.9476** |
| **Balanced Accuracy** | 0.7056 | **0.9525** |
| **Recall (Clase Churn = 1)** | 0.4242 | **0.9596** |
| **Precision (Clase Churn = 1)** | 0.8571 | **0.7661** |
| **F1-Score Ponderado** | 0.8845 | **0.9499** |
| **Área Bajo la Curva ROC (AUC)** | 0.9234 | **0.9850** |

### 6.3 Justificación Técnica de la Elección
Debido a la asimetría en los costos de error asociados al problema de negocio, no identificar a un cliente que efectivamente abandonará el servicio (falso negativo) conlleva la pérdida del valor de vida del cliente (*lifetime value*), mientras que un falso positivo implica únicamente una acción preventiva de contacto. Por consiguiente, el modelo **Random Forest** fue seleccionado por su sobresaliente capacidad de captura (**Recall de 95,96%** frente a 42,42% del modelo lineal) y un balance global superior (**F1 ponderado de 0.9499**).

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
│   ├── app.py               # Aplicación interactiva de servicio Streamlit
│   ├── build_notebook.py    # Generador automatizado del cuaderno unificado
│   ├── ml_core.py           # Módulo central: inferencia de tipos, pipelines y métricas
│   └── train.py             # Script de entrenamiento por línea de comandos
├── .dockerignore            # Reglas de exclusión para el contexto de Docker
├── .gitignore               # Reglas de exclusión para Git
├── app.py                   # Enlace de servicio en la raíz del contenedor
├── Dockerfile               # Instrucciones de construcción de la imagen de producción
├── ML_Supervisado_Streamlit_Docker_GCP.ipynb # Cuaderno ejecutable en Google Colab
├── ml_core.py               # Módulo central en raíz para el contenedor
├── model.joblib             # Artefacto serializado (modelo entrenado + metadatos)
├── README.md                # Documentación técnica institucional
├── requirements.txt         # Especificación de dependencias con versiones compatibles
└── train.py                 # Script de entrenamiento en raíz para el contenedor
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
  "Call Failure": 8.0,
  "Subscription Length": 38.0,
  "Charge Amount": 0.0,
  "Seconds of Use": 4370.0,
  "Frequency of use": 71.0,
  "Frequency of SMS": 5.0,
  "Distinct Called Numbers": 17.0,
  "Complains": "0",
  "Age Group": "3",
  "Tariff Plan": "1",
  "Status": "1",
  "Age": "30"
}
```

### 11.2 Salida Generada por el Modelo
- **Clase Predicha:** `0` (Permanencia / No Abandono).
- **Probabilidades Estimadas:** 
  - Probabilidad de Permanencia (`P(Y=0)`): 0.9862
  - Probabilidad de Abandono (`P(Y=1)`): 0.0138

---

## 12. Protocolo de Auditoría y Prueba Cruzada

Para validar la reproducibilidad externa requerida en la sección 31 de la guía, un evaluador independiente debe poder operar el sistema siguiendo exclusivamente los siguientes pasos:

1. Clonar el repositorio mediante la URL pública.
2. Comprobar la integridad de los archivos raíz (`Dockerfile`, `requirements.txt`, `app.py`, `model.joblib`).
3. Ejecutar el comando `docker build -t ml-streamlit-app .` sin alertas de falla de dependencias.
4. Ejecutar el contenedor mediante `docker run -d -p 8501:8501 ml-streamlit-app`.
5. Enviar una petición a través de la pestaña de predicción unitaria o predicción por lotes y verificar la respuesta esperada.

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
  *Programa de Maestría en Inteligencia Artificial / Ciencia de Datos*  
  *Repositorio Académico:* [julianjimenez4809-ui](https://github.com/julianjimenez4809-ui)
