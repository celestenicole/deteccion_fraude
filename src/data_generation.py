"""
Cliente para AWS Bedrock - Generación de descripciones crediticias
"""
import boto3
import json
import pandas as pd
import time
from typing import List, Dict, Optional
from botocore.exceptions import ClientError
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockClient:
    """
    Cliente para interactuar con AWS Bedrock para generación de contenido
    """
    
    def __init__(self, region_name: str = "us-east-1", model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        """
        Inicializar cliente Bedrock
        
        Args:
            region_name: Región de AWS
            model_id: ID del modelo de Bedrock a usar
        """
        self.region_name = region_name
        self.model_id = model_id
        
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region_name
            )
            logger.info(f"✅ Cliente Bedrock inicializado en región: {region_name}")
            logger.info(f"🤖 Modelo configurado: {model_id}")
        except Exception as e:
            logger.error(f"❌ Error al inicializar Bedrock: {e}")
            raise
    
    def generate_credit_description(self, customer_data: Dict) -> str:
        """
        Genera una descripción narrativa de un cliente usando Bedrock
        
        Args:
            customer_data: Diccionario con datos del cliente
            
        Returns:
            Descripción generada como string
        """
        
        # Crear prompt estructurado
        prompt = self._create_description_prompt(customer_data)
        
        try:
            # Preparar request para Claude
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.1,
                 "top_k": 250,
    "top_p": 0.9,
    
  "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Llamar a Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            
            # Procesar respuesta
            response_body = json.loads(response['body'].read())
            description = response_body['content'][0]['text'].strip()
            
            logger.info(f"✅ Descripción generada exitosamente")
            return description
            
        except ClientError as e:
            logger.error(f"❌ Error de AWS: {e}")
            return f"Error al generar descripción: {str(e)}"
        except Exception as e:
            logger.error(f"❌ Error inesperado: {e}")
            return f"Error inesperado: {str(e)}"
    
    def classify_credit_risk(self, customer_data: Dict, description: str) -> Dict:
        """
        Clasifica el riesgo crediticio usando Bedrock
        
        Args:
            customer_data: Datos del cliente
            description: Descripción del cliente
            
        Returns:
            Diccionario con clasificación y confianza
        """
        
        prompt = self._create_classification_prompt(customer_data, description)
        
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "temperature": 0.1,
                 "top_k": 250,
    "top_p": 0.9,
    

                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            classification_text = response_body['content'][0]['text'].strip()
            
            # Parsear la respuesta
            result = self._parse_classification_response(classification_text)
            
            logger.info(f"✅ Clasificación completada: {result['prediction']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en clasificación: {e}")
            return {
                "prediction": "unknown",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}"
            }
    
    def _create_description_prompt(self, data: Dict) -> str:
       return f"""
    As a credit analyst, write a concise and professional 2-paragraph description of this client based on the data.

Client data:
{json.dumps(data)}

Include: demographic info, financial profile, and purpose of credit.
Be objective, no opinions.
"""
    
    def _create_classification_prompt(self, data: Dict, description: str) -> str:
        return f"""
    You are a credit analyst. Based on the description and client data, classify the credit risk.

Description:
\"\"\"{description}\"\"\"

Client data:
{json.dumps(data)}

Respond in JSON format:
{{
  "prediction": "GOOD or BAD",
  "confidence": float (between 0.0 and 1.0),
  "reasoning": "short explanation"
}}
"""
    
    def _parse_classification_response(self, response_text: str) -> Dict:
        """
        Parsea la respuesta de clasificación
        """
        try:
            # Buscar JSON en la respuesta
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_text = response_text[start_idx:end_idx]
                result = json.loads(json_text)
                
                # Validar campos requeridos
                if 'prediction' in result and 'confidence' in result:
                    return result
            
            # Si no se puede parsear, usar clasificación básica
            if 'good' in response_text.lower():
                return {"prediction": "good", "confidence": 0.6, "reasoning": "Parsed from text"}
            elif 'bad' in response_text.lower():
                return {"prediction": "bad", "confidence": 0.6, "reasoning": "Parsed from text"}
            else:
                return {"prediction": "unknown", "confidence": 0.0, "reasoning": "Could not parse"}
                
        except Exception as e:
            logger.error(f"Error parsing classification: {e}")
            return {"prediction": "unknown", "confidence": 0.0, "reasoning": f"Parse error: {str(e)}"}
    
    def process_batch(self, df: pd.DataFrame, batch_size: int = 5) -> pd.DataFrame:
        """
        Procesa un lote de registros
        
        Args:
            df: DataFrame con datos de clientes
            batch_size: Tamaño del lote
            
        Returns:
            DataFrame enriquecido con descripciones y clasificaciones
        """
        logger.info(f"🔄 Procesando {len(df)} registros en lotes de {batch_size}")
        
        results = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            logger.info(f"📦 Procesando lote {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            
            for idx, row in batch.iterrows():
                try:
                    # Convertir fila a diccionario
                    customer_data = row.to_dict()
                    
                    # Generar descripción
                    description = self.generate_credit_description(customer_data)
                    
                    # Clasificar riesgo
                    classification = self.classify_credit_risk(customer_data, description)
                    
                    # Agregar resultados
                    result = customer_data.copy()
                    result['bedrock_description'] = description
                    result['bedrock_prediction'] = classification['prediction']
                    result['bedrock_confidence'] = classification['confidence']
                    result['bedrock_reasoning'] = classification['reasoning']
                    
                    results.append(result)
                    
                    # Pausa para evitar rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"❌ Error procesando fila {idx}: {e}")
                    # Agregar fila con errores
                    result = row.to_dict()
                    result['bedrock_description'] = f"Error: {str(e)}"
                    result['bedrock_prediction'] = "error"
                    result['bedrock_confidence'] = 0.0
                    result['bedrock_reasoning'] = f"Processing error: {str(e)}"
                    results.append(result)
        
        enriched_df = pd.DataFrame(results)
        logger.info(f"✅ Procesamiento completado: {len(enriched_df)} registros")
        
        return enriched_df

if __name__ == "__main__":
    # Test básico
    client = BedrockClient()
    
    test_data = {
        "age": 35,
        "sex": "male",
        "job": "skilled",
        "housing": "own",
        "credit_amount": 5000,
        "duration": 24,
        "purpose": "car"
    }
    
    print("🧪 Probando generación de descripción...")
    description = client.generate_credit_description(test_data)
    print(f"📝 Descripción: {description}")
    
    print("\n🧪 Probando clasificación...")
    classification = client.classify_credit_risk(test_data, description)
    print(f"🎯 Clasificación: {classification}")
