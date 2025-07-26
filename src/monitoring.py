
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path

class CreditRiskMonitor:
    def __init__(self, log_dir: str = "../logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'credit_risk_api.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('CreditRiskMonitor')

        # Métricas en memoria
        self.metrics = {
            'predictions': [],
            'response_times': [],
            'ml_predictions': [],
            'bedrock_predictions': [],
            'agreements': [],
            'errors': []
        }

    def log_prediction(self, request_data: Dict, ml_result: str, ml_prob: float,
                      bedrock_result: str, bedrock_conf: float, response_time: float):
        """Registrar una predicción para monitoreo"""

        prediction_log = {
            'timestamp': datetime.now().isoformat(),
            'request': request_data,
            'ml_prediction': ml_result,
            'ml_probability': ml_prob,
            'bedrock_prediction': bedrock_result,
            'bedrock_confidence': bedrock_conf,
            'agreement': ml_result == bedrock_result,
            'response_time': response_time
        }

        # Agregar a métricas
        self.metrics['predictions'].append(prediction_log)
        self.metrics['response_times'].append(response_time)
        self.metrics['ml_predictions'].append(ml_result)
        self.metrics['bedrock_predictions'].append(bedrock_result)
        self.metrics['agreements'].append(ml_result == bedrock_result)

        # Log
        self.logger.info(f"Predicción: ML={ml_result}({ml_prob:.2f}), "
                        f"Bedrock={bedrock_result}({bedrock_conf:.2f}), "
                        f"Acuerdo={ml_result == bedrock_result}, RT={response_time:.2f}s")

        # Guardar en archivo
        self._save_prediction_log(prediction_log)

        # Verificar alertas
        self._check_alerts()

    def log_error(self, error_type: str, error_message: str, request_data: Dict = None):
        """Registrar un error"""

        error_log = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'request_data': request_data
        }

        self.metrics['errors'].append(error_log)
        self.logger.error(f"Error {error_type}: {error_message}")

        # Guardar error
        with open(self.log_dir / 'errors.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(error_log, ensure_ascii=False) + '\n')

    def get_metrics_summary(self) -> Dict:
        """Obtener resumen de métricas"""

        if not self.metrics['predictions']:
            return {'message': 'No hay datos de predicciones aún'}

        # Calcular métricas
        total_predictions = len(self.metrics['predictions'])
        agreement_rate = sum(self.metrics['agreements']) / total_predictions * 100
        avg_response_time = sum(self.metrics['response_times']) / len(self.metrics['response_times'])

        # Distribución de predicciones
        ml_good = self.metrics['ml_predictions'].count('good')
        ml_bad = self.metrics['ml_predictions'].count('bad')
        bedrock_good = self.metrics['bedrock_predictions'].count('good')
        bedrock_bad = self.metrics['bedrock_predictions'].count('bad')

        # Últimas 10 predicciones
        recent_predictions = self.metrics['predictions'][-10:]

        return {
            'total_predictions': total_predictions,
            'agreement_rate': agreement_rate,
            'avg_response_time': avg_response_time,
            'ml_distribution': {'good': ml_good, 'bad': ml_bad},
            'bedrock_distribution': {'good': bedrock_good, 'bad': bedrock_bad},
            'total_errors': len(self.metrics['errors']),
            'recent_predictions': recent_predictions,
            'status': self._get_system_status()
        }

    def _save_prediction_log(self, prediction_log: Dict):
        """Guardar log de predicción en archivo"""
        with open(self.log_dir / 'predictions.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(prediction_log, ensure_ascii=False) + '\n')

    def _check_alerts(self):
        """Verificar condiciones de alerta"""

        if len(self.metrics['predictions']) < 10:
            return  # Necesitamos al menos 10 predicciones

        # Últimas 10 predicciones
        recent = self.metrics['predictions'][-10:]
        recent_agreements = [p['agreement'] for p in recent]
        recent_times = [p['response_time'] for p in recent]

        # Alertas
        agreement_rate = sum(recent_agreements) / len(recent_agreements) * 100
        avg_time = sum(recent_times) / len(recent_times)

        if agreement_rate < 70:
            self.logger.warning(f"🚨 ALERTA: Concordancia baja entre modelos: {agreement_rate:.1f}%")

        if avg_time > 5:
            self.logger.warning(f"🚨 ALERTA: Tiempo de respuesta alto: {avg_time:.2f}s")

        # Contar errores recientes (última hora)
        recent_errors = [e for e in self.metrics['errors'] 
                        if (datetime.now() - datetime.fromisoformat(e['timestamp'])).seconds < 3600]

        if len(recent_errors) > 5:
            self.logger.warning(f"🚨 ALERTA: Muchos errores recientes: {len(recent_errors)}")

    def _get_system_status(self) -> str:
        """Determinar el estado del sistema"""

        if not self.metrics['predictions']:
            return "INICIANDO"

        # Últimas métricas
        recent_agreements = self.metrics['agreements'][-10:] if len(self.metrics['agreements']) >= 10 else self.metrics['agreements']
        recent_times = self.metrics['response_times'][-10:] if len(self.metrics['response_times']) >= 10 else self.metrics['response_times']
        recent_errors = len([e for e in self.metrics['errors'] 
                           if (datetime.now() - datetime.fromisoformat(e['timestamp'])).seconds < 3600])

        if recent_errors > 5:
            return "CRÍTICO"
        elif len(recent_agreements) > 0 and sum(recent_agreements) / len(recent_agreements) < 0.7:
            return "ADVERTENCIA"
        elif len(recent_times) > 0 and sum(recent_times) / len(recent_times) > 5:
            return "DEGRADADO"
        else:
            return "SALUDABLE"

# Inicializar monitor global
monitor = CreditRiskMonitor()
