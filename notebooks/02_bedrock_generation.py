# %% md
# # 🧠 AWS Bedrock – Generative AI for Hybrid System
#
# This notebook implements the integration with **AWS Bedrock (Claude 3 Haiku)** as part of a hybrid credit risk detection system.
#
# ## 📈 Objectives (IMPLEMENTED):
# 1. **✔️ Generate enriched customer descriptions** using Generative AI
# 2. **✔️ Classify credit risk** using Claude 3 Haiku
# 3. **✔️ Create enriched dataset** for ML training
# 4. **✔️ Integrate with hybrid API** ML + Generative AI
#
# ## 🎯 **ROLE IN THE FINAL SYSTEM:**
# Bedrock acts as the **second model** in the hybrid system, providing:
# - Detailed risk explanations
# - Cross-validation with traditional ML
# - Advanced reasoning capabilities

# %%
# Install dependencies if needed
# !pip install boto3 pandas numpy python-dotenv

# %%
import boto3
import json
import pandas as pd
import numpy as np
import time
from pathlib import Path
import os
from typing import Dict, List
from botocore.exceptions import ClientError
import logging
from IPython.display import display

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("✔️ Libraries successfully imported")

# %%
# AWS configuration
AWS_REGION = "us-east-1"
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

print(f"🛠️ Configured region: {AWS_REGION}")
print(f"🧠 Bedrock model: {BEDROCK_MODEL_ID}")
print("\n⚠️ IMPORTANT: Make sure your AWS credentials are configured")
print("   - Run: aws configure")
print("   - Ensure AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set")

# %%
# Load real challenge data
data_path = Path("../data/credir_risk_reto.xlsx")

print(f"📂 Loading data from: {data_path}")

if data_path.exists():
    df = pd.read_excel(data_path)
    print("✔️ Data loaded successfully")
    print(f"📈 Dataset shape: {df.shape}")
    print(f"🗒️ Columns: {list(df.columns)}")

    # Save as CSV for later use
    csv_path = "../data/credit_risk_reto.csv"
    df.to_csv(csv_path, index=False)
    print(f"📁 Saved as CSV: {csv_path}")

    # Show sample
    display(df.head())
else:
    print(f"✖️ File not found: {data_path}")
    print("Please make sure the file is in the data folder/")
