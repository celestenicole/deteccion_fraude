
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import sys
import os

# Agregar el directorio src al path
sys.path.append('../src')
from data_generation import BedrockClient

app = FastAPI(title="Credit Risk API", description="API para predicción de riesgo crediticio")

# Cargar modelos
try:
    rf_model = joblib.load('../models/local_rf_model.pkl')
    label_encoders = joblib.load('../models/label_encoders.pkl')
    bedrock_client = BedrockClient()
    print("✅ Modelos cargados exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")
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

@app.get("/")
async def root():
    return {"message": "Credit Risk Prediction API", "status": "running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_credit_risk(request: CreditRequest):
    try:
        if rf_model is None or bedrock_client is None:
            raise HTTPException(status_code=500, detail="Modelos no disponibles")

        # Preparar datos para ML
        data = request.dict()

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

        # Recomendación final (combinando ambos modelos)
        if ml_pred == "bad" and bedrock_result['prediction'] == "bad":
            recommendation = "RECHAZAR - Ambos modelos predicen alto riesgo"
        elif ml_pred == "good" and bedrock_result['prediction'] == "good":
            recommendation = "APROBAR - Ambos modelos predicen bajo riesgo"
        else:
            recommendation = "REVISAR MANUALMENTE - Modelos discrepan"

        return PredictionResponse(
            ml_prediction=ml_pred,
            ml_probability=float(ml_proba),
            bedrock_prediction=bedrock_result['prediction'],
            bedrock_confidence=bedrock_result['confidence'],
            bedrock_reasoning=bedrock_result['reasoning'],
            recommendation=recommendation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "ml_model": rf_model is not None,
        "bedrock_client": bedrock_client is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
