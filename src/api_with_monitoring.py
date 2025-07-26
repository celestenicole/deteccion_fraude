
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime

# Agregar el directorio src al path
sys.path.append('../src')
from src.data_generation import BedrockClient
from src.monitoring import CreditRiskMonitor

app = FastAPI(
    title="Credit Risk Detection API",
    description="API para predicción de riesgo crediticio con monitoreo en tiempo real",
    version="1.0.0"
)

# Inicializar componentes
monitor = CreditRiskMonitor()

# Cargar modelos
try:
    rf_model = joblib.load('C:/Users/celes/OneDrive/Escritorio/Reto_detección_fraude/models/local_rf_model.pkl')
    label_encoders = joblib.load('C:/Users/celes/OneDrive/Escritorio/Reto_detección_fraude/models/label_encoders.pkl')
    bedrock_client = BedrockClient()
    monitor.logger.info("✅ Modelos cargados exitosamente")
except Exception as e:
    monitor.log_error("MODEL_LOADING", str(e))
    rf_model = None
    label_encoders = None
    bedrock_client = None

class CreditRequest(BaseModel):
    age: int
    sex: str
    job: int
    housing: str
    saving_accounts: str = None
    checking_account: str = None
    credit_amount: float
    duration: int
    purpose: str

class PredictionResponse(BaseModel):
    ml_prediction: str
    ml_probability: float
    bedrock_prediction: str
    bedrock_confidence: float
    bedrock_reasoning: str
    recommendation: str
    response_time: float
    timestamp: str

@app.get("/")
async def root():
    return {
        "message": "Credit Risk Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_credit_risk(request: CreditRequest):
    start_time = time.time()
    request_data = request.dict()

    try:
        if rf_model is None or bedrock_client is None:
            monitor.log_error("MODEL_UNAVAILABLE", "Modelos no disponibles", request_data)
            raise HTTPException(status_code=500, detail="Modelos no disponibles")

        # Preparar datos para ML
        data = request_data.copy()

        # Encoding de variables categóricas
        categorical_columns = ['sex', 'housing', 'saving_accounts', 'checking_account', 'purpose']
        for col in categorical_columns:
            if col in data and col in label_encoders:
                value = data[col] if data[col] is not None else 'unknown'
                try:
                    data[f'{col}_encoded'] = label_encoders[col].transform([value])[0]
                except:
                    data[f'{col}_encoded'] = 0  # Valor desconocido

        # Crear features para ML
        features = [
            data['age'], data['credit_amount'], data['duration'],
            data.get('sex_encoded', 0), data.get('job', 0), 
            data.get('housing_encoded', 0), data.get('saving_accounts_encoded', 0),
            data.get('checking_account_encoded', 0), data.get('purpose_encoded', 0)
        ]

        # Predicción ML
        ml_proba = rf_model.predict_proba([features])[0][1]
        ml_pred = "bad" if ml_proba > 0.5 else "good"

        # Predicción Bedrock
        customer_data = {
            "age": request.age,
            "sex": request.sex,
            "job": request.job,
            "housing": request.housing,
            "credit_amount": request.credit_amount,
            "duration": request.duration,
            "purpose": request.purpose
        }

        # Generar descripción con Bedrock
        description = bedrock_client.generate_credit_description(customer_data)

        # Clasificar con Bedrock
        bedrock_result = bedrock_client.classify_credit_risk(customer_data, description)

        # Recomendación final
        if ml_pred == "bad" and bedrock_result['prediction'] == "bad":
            recommendation = "RECHAZAR - Ambos modelos predicen alto riesgo"
        elif ml_pred == "good" and bedrock_result['prediction'] == "good":
            recommendation = "APROBAR - Ambos modelos predicen bajo riesgo"
        else:
            recommendation = "REVISAR MANUALMENTE - Modelos discrepan"

        # Calcular tiempo de respuesta
        response_time = time.time() - start_time

        # Registrar en monitoreo
        monitor.log_prediction(
            request_data, ml_pred, ml_proba,
            bedrock_result['prediction'], bedrock_result['confidence'],
            response_time
        )

        return PredictionResponse(
            ml_prediction=ml_pred,
            ml_probability=float(ml_proba),
            bedrock_prediction=bedrock_result['prediction'],
            bedrock_confidence=bedrock_result['confidence'],
            bedrock_reasoning=bedrock_result['reasoning'],
            recommendation=recommendation,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        response_time = time.time() - start_time
        monitor.log_error("PREDICTION_ERROR", str(e), request_data)
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "ml_model": rf_model is not None,
            "bedrock_client": bedrock_client is not None
        },
        "system_status": monitor._get_system_status()
    }

@app.get("/metrics")
async def get_metrics():
    """Endpoint para obtener métricas del sistema"""
    return monitor.get_metrics_summary()

if __name__ == "__main__":
    import uvicorn
    monitor.logger.info("🚀 Iniciando Credit Risk API con monitoreo")
    uvicorn.run(app, host="0.0.0.0", port=8000)
