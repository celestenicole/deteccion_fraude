# 🛡 Detección de Fraudes en Transacciones Bancarias con AWS Bedrock y SageMaker

## 📌 Descripción del Proyecto

Este proyecto tiene como objetivo detectar fraudes en transacciones bancarias utilizando tecnologías de Inteligencia Artificial y Aprendizaje Automático, integrando *modelos generativos de AWS Bedrock* (como GPT o Llama) para generar descripciones y etiquetas, y un modelo supervisado entrenado y desplegado en *Amazon SageMaker* para clasificar el riesgo crediticio como “good risk” o “bad risk”.

---

## 🗂 Dataset Utilizado

El conjunto de datos utilizado es credit_risk_reto.csv, que contiene variables relacionadas al riesgo crediticio de clientes. Se encuentra en la carpeta /data.

---

## 🔁 Flujo del Proyecto

1. *Análisis Exploratorio de Datos (EDA)*  
   Se analizan valores nulos, distribuciones, correlaciones y estadísticas descriptivas.

2. *Generación de Descripciones con AWS Bedrock*  
   Se utilizan modelos generativos para crear descripciones en lenguaje natural sobre cada registro.

3. *Clasificación de Riesgo con Bedrock*  
   A partir de las descripciones generadas, se clasifican los casos en good risk o bad risk.

4. *Entrenamiento de Modelo Supervisado con SageMaker*  
   Se entrena un modelo supervisado (Random Forest) usando las etiquetas generadas.

5. *Despliegue del Modelo y Monitoreo*  
   El modelo es desplegado localmente (simulación de endpoint) y monitoreado desde un servicio FastAPI.

---

## 🖥 Requisitos de Instalación

Asegúrate de tener instalado:

- Python 3.11 o superior
- pip
- Visual Studio Code
- Git (opcional)

### 🔧 Instalación de dependencias

Activa el entorno virtual:

```bash
.\venv\Scripts\activate
