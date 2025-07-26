
import requests
import json
import time

# URL de la API (cambiar si está en otro puerto/host)
API_URL = "http://localhost:8000"

def test_prediction():
    """Test de predicción con datos de ejemplo"""

    # Datos de prueba
    test_data = {
        "age": 35,
        "sex": "male",
        "job": 2,
        "housing": "own",
        "saving_accounts": "little",
        "checking_account": "moderate",
        "credit_amount": 5000.0,
        "duration": 24,
        "purpose": "car"
    }

    try:
        print("🔍 Probando predicción...")
        response = requests.post(f"{API_URL}/predict", json=test_data)

        if response.status_code == 200:
            result = response.json()
            print("✅ Predicción exitosa!")
            print(f"ML Predicción: {result['ml_prediction']} (Prob: {result['ml_probability']:.2f})")
            print(f"Bedrock Predicción: {result['bedrock_prediction']} (Confianza: {result['bedrock_confidence']:.2f})")
            print(f"Recomendación: {result['recommendation']}")
            print(f"Razonamiento: {result['bedrock_reasoning'][:100]}...")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ No se puede conectar a la API. ¿Está ejecutándose?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health():
    """Test del endpoint de salud"""
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ API saludable!")
            print(f"Estado: {health['status']}")
            print(f"Modelo ML: {'✅' if health['ml_model'] else '❌'}")
            print(f"Cliente Bedrock: {'✅' if health['bedrock_client'] else '❌'}")
            return True
        else:
            print(f"❌ Error en health check: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en health check: {e}")
        return False

def run_tests():
    """Ejecutar todos los tests"""
    print("🧪 Iniciando tests de la API...")
    print("=" * 50)

    # Test de salud
    print("1. Health Check:")
    health_ok = test_health()
    print()

    # Test de predicción
    print("2. Test de Predicción:")
    prediction_ok = test_prediction()
    print()

    # Resumen
    print("=" * 50)
    print("📊 Resumen de Tests:")
    print(f"Health Check: {'✅' if health_ok else '❌'}")
    print(f"Predicción: {'✅' if prediction_ok else '❌'}")

    if health_ok and prediction_ok:
        print("🎉 ¡Todos los tests pasaron!")
    else:
        print("⚠️ Algunos tests fallaron")

if __name__ == "__main__":
    run_tests()
