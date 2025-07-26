#%% md
# # 📊 Exploratory Data Analysis - Credit Risk
#
# This notebook performs an initial analysis of the dataset for the credit risk detection challenge.
#%%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Visualization settings
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)

print("✅ Libraries successfully imported")
#%%
# Load dataset
data_path = Path("../data/credir_risk_reto.xlsx")
print(f"📂 Loading data from: {data_path}")

df = pd.read_excel(data_path)
print("✅ Data loaded successfully")
print(f"📊 Dataset dimensions: {df.shape}")
#%%
# General dataset overview
print("🔍 GENERAL DATASET OVERVIEW")
print("=" * 50)
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print("\nAvailable columns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")
#%%
# Data types and missing values
print("📋 DATA TYPES AND MISSING VALUES")
print("=" * 50)

info_df = pd.DataFrame({
    'Data_Type': df.dtypes,
    'Missing_Values': df.isnull().sum(),
    'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
    'Unique_Values': df.nunique()
})

display(info_df)
#%%
# Preview the first rows
print("👁️ FIRST 10 ROWS OF THE DATASET")
print("=" * 50)
display(df.head(10))
#%%
# Descriptive statistics
print("📈 DESCRIPTIVE STATISTICS")
print("=" * 50)
display(df.describe(include='all'))
#%%
# Identify the target variable
print("🎯 TARGET VARIABLE IDENTIFICATION")
print("=" * 50)

# Search for columns that could be targets
target_candidates = []
for col in df.columns:
    unique_vals = df[col].nunique()
    if unique_vals <= 5:  # Few unique values
        print(f"{col}: {unique_vals} unique values -> {df[col].unique()}")
        target_candidates.append(col)

print(f"\n🎯 Potential target candidates: {target_candidates}")
#%%
# Save dataset as CSV for further use
csv_path = "../data/credit_risk_reto.csv"
df.to_csv(csv_path, index=False)
print(f"✅ Dataset saved as CSV: {csv_path}")

# Display final summary
print("\n📊 FINAL SUMMARY:")
print(f"• Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"• CSV file created: {csv_path}")
print("• Ready to proceed with AWS Bedrock 🚀")
#%%
