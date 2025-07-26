"""
Configuración global del proyecto
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Crear directorios si no existen
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Configuración AWS
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Configuración SageMaker
SAGEMAKER_ROLE = os.getenv("SAGEMAKER_ROLE")
S3_BUCKET = os.getenv("S3_BUCKET")

# Configuración Bedrock
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-east-1")

# Configuración del proyecto
PROJECT_NAME = os.getenv("PROJECT_NAME", "credit-risk-detection")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-risk-model")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "credit-risk-endpoint")

# Configuración de desarrollo
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Archivos de datos
ORIGINAL_DATA_FILE = DATA_DIR / "credit_risk_reto.csv"
ENRICHED_DATA_FILE = DATA_DIR / "credit_risk_enriched.csv"

# Configuración de Bedrock
BEDROCK_CONFIG = {
    "model_id": BEDROCK_MODEL_ID,
    "region": BEDROCK_REGION,
    "max_tokens": 1000,
    "temperature": 0.1,
    "timeout": 60
}

# Configuración de SageMaker
SAGEMAKER_CONFIG = {
    "role": SAGEMAKER_ROLE,
    "instance_type": "ml.m5.large",
    "instance_count": 1,
    "max_runtime_in_seconds": 3600
}

print(f"✅ Configuración cargada para proyecto: {PROJECT_NAME}")
if DEBUG:
    print(f"📂 Directorio de datos: {DATA_DIR}")
    print(f"🔧 Región AWS: {AWS_REGION}")
    print(f"🤖 Modelo Bedrock: {BEDROCK_MODEL_ID}")
