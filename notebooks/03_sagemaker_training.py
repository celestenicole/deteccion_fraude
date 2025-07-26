# Initial configuration
import pandas as pd
import numpy as np
import boto3
import sagemaker
from sagemaker import get_execution_role
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

print("📦 Libraries successfully imported")
print(f"🔗 SageMaker version: {sagemaker.__version__}")

# Configure SageMaker
try:
    role = get_execution_role()
    print(f"✔️ SageMaker role: {role}")
except:
    # Local development fallback
    role = "arn:aws:iam::075664900662:role/SageMakerExecutionRole"
    print(f"⚠️ Using default role: {role}")

# Create SageMaker session
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
region = boto3.Session().region_name

print(f"🪣 S3 Bucket: {bucket}")
print(f"🌍 Region: {region}")

# Load data
print("\n📈 Loading data...")
try:
    df_enriched = pd.read_csv('../data/credit_risk_enriched.csv')
    print(f"✔️ Enriched data loaded: {df_enriched.shape}")
    use_enriched = True
except:
    df_enriched = pd.read_csv('../data/credit_risk_reto.csv')
    print(f"⚠️ Using original data: {df_enriched.shape}")
    use_enriched = False

df_enriched.head()


# Enriched data preprocessing
def prepare_training_data(df):
    """Prepare data for ML training"""

    df_processed = df.copy()

    if 'target' not in df_processed.columns:
        if 'bedrock_prediction' in df_processed.columns:
            df_processed['target'] = (df_processed['bedrock_prediction'] == 'bad').astype(int)
            print("✔️ Target created from Bedrock classification")
        else:
            np.random.seed(42)
            risk_score = (df_processed['Age'] < 25).astype(int) * 0.3 + \
                         (df_processed['Credit amount'] > df_processed['Credit amount'].quantile(0.8)).astype(
                             int) * 0.4 + \
                         (df_processed['Duration'] > 24).astype(int) * 0.3
            df_processed['target'] = (risk_score > 0.5).astype(int)
            print("✔️ Synthetic target created based on business rules")

    label_encoders = {}
    categorical_columns = ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']

    for col in categorical_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('unknown')
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            label_encoders[col] = le

    feature_columns = ['Age', 'Credit amount', 'Duration'] + \
                      [f'{col}_encoded' for col in categorical_columns if col in df_processed.columns]

    if 'bedrock_confidence' in df_processed.columns:
        feature_columns.append('bedrock_confidence')
        print("✔️ Including Bedrock confidence as a feature")

    X = df_processed[feature_columns].fillna(0)
    y = df_processed['target']

    print(f"📈 Dataset ready: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")

    return X, y, feature_columns, label_encoders


X, y, feature_columns, label_encoders = prepare_training_data(df_enriched)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✔️ Data split completed:")
print(f"📈 Training: {X_train.shape}")
print(f"🧪 Testing: {X_test.shape}")

print(f"\n🛠️ Features used: {feature_columns}")

# Local training (cheaper)
print("🏠 Training model locally...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

y_pred_local = rf_model.predict(X_test)
y_pred_proba_local = rf_model.predict_proba(X_test)[:, 1]

accuracy_local = accuracy_score(y_test, y_pred_local)
print(f"🎯 Local model accuracy: {accuracy_local:.3f}")
print(f"📈 Classification Report:")
print(classification_report(y_test, y_pred_local))

os.makedirs('../models', exist_ok=True)
joblib.dump(rf_model, '../models/local_rf_model.pkl')
joblib.dump(label_encoders, '../models/label_encoders.pkl')
print("📁 Local model saved to ../models/")
