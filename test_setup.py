"""
Script de prueba para verificar la configuración y datos
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

def test_data_loading():
    """
    Prueba cargar los datos reales del reto
    """
    print("🔍 PROBANDO CARGA DE DATOS")
    print("=" * 50)
    
    # Buscar archivo Excel
    data_path = Path("data/credir_risk_reto.xlsx")
    
    if data_path.exists():
        print(f"✅ Archivo encontrado: {data_path}")
        
        # Cargar datos
        try:
            df = pd.read_excel(data_path)
            print(f"✅ Datos cargados exitosamente")
            print(f"📊 Forma del dataset: {df.shape}")
            print(f"📋 Columnas: {list(df.columns)}")
            
            # Información básica
            print(f"\n🔍 Información básica:")
            print(f"• Filas: {len(df)}")
            print(f"• Columnas: {len(df.columns)}")
            print(f"• Tipos de datos: {df.dtypes.value_counts().to_dict()}")
            
            # Valores nulos
            print(f"\n❌ Valores nulos:")
            null_counts = df.isnull().sum()
            has_nulls = False
            for col, count in null_counts.items():
                if count > 0:
                    print(f"• {col}: {count} ({count/len(df)*100:.1f}%)")
                    has_nulls = True
            if not has_nulls:
                print("• ✅ No hay valores nulos")
            
            # Muestra de datos
            print(f"\n📋 Primeras 3 filas:")
            print(df.head(3).to_string())
            
            # Guardar como CSV
            csv_path = "data/credit_risk_reto.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n💾 Datos guardados como CSV: {csv_path}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return None
    else:
        print(f"❌ Archivo no encontrado: {data_path}")
        print("📁 Archivos en directorio data/:")
        data_dir = Path("data")
        if data_dir.exists():
            for file in data_dir.iterdir():
                print(f"  - {file.name}")
        return None

def test_aws_boto3():
    """
    Prueba importar boto3 y verificar configuración básica
    """
    print(f"\n🔍 PROBANDO CONFIGURACIÓN AWS")
    print("=" * 50)
    
    try:
        import boto3
        print("✅ boto3 importado correctamente")
        
        # Intentar crear un cliente (no hacer requests aún)
        try:
            session = boto3.Session()
            region = session.region_name or "us-east-1"
            print(f"✅ Sesión AWS creada, región: {region}")
            
            # Verificar credenciales (sin hacer requests)
            credentials = session.get_credentials()
            if credentials:
                print("✅ Credenciales AWS encontradas")
                return True
            else:
                print("⚠️ Credenciales AWS no configuradas")
                print("🔧 Para configurar:")
                print("   aws configure")
                print("   O establecer variables de entorno:")
                print("   - AWS_ACCESS_KEY_ID")
                print("   - AWS_SECRET_ACCESS_KEY")
                return False
                
        except Exception as e:
            print(f"⚠️ Error configuración AWS: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ Error importando boto3: {e}")
        print("🔧 Instalar con: pip install boto3")
        return False

def test_environment():
    """
    Verifica el entorno de desarrollo
    """
    print(f"\n🔍 VERIFICANDO ENTORNO")
    print("=" * 50)
    
    import sys
    print(f"🐍 Python: {sys.version}")
    print(f"📁 Directorio actual: {os.getcwd()}")
    
    # Verificar librerías clave
    libraries = ["pandas", "numpy", "boto3", "matplotlib", "seaborn"]
    for lib in libraries:
        try:
            __import__(lib)
            print(f"✅ {lib} disponible")
        except ImportError:
            print(f"❌ {lib} no disponible")

def main():
    """
    Función principal de pruebas
    """
    print("🚀 INICIANDO PRUEBAS DEL PROYECTO")
    print("=" * 60)
    
    # 1. Verificar entorno
    test_environment()
    
    # 2. Cargar datos
    df = test_data_loading()
    
    # 3. Verificar AWS
    aws_ok = test_aws_boto3()
    
    # Resumen
    print(f"\n📋 RESUMEN")
    print("=" * 50)
    print(f"✅ Datos: {'OK' if df is not None else 'ERROR'}")
    print(f"{'✅' if aws_ok else '⚠️'} AWS: {'OK' if aws_ok else 'Configurar credenciales'}")
    
    if df is not None and aws_ok:
        print(f"\n🎉 ¡Todo listo para continuar!")
        print(f"📝 Próximos pasos:")
        print(f"   1. Ejecutar notebook: notebooks/01_exploratory_analysis.ipynb")
        print(f"   2. Ejecutar notebook: notebooks/02_bedrock_generation.ipynb")
    elif df is not None:
        print(f"\n⚠️ Datos OK, pero configurar AWS antes de usar Bedrock")
    else:
        print(f"\n❌ Resolver problemas antes de continuar")

if __name__ == "__main__":
    main()
